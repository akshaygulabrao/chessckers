"""PUCT MCTS for Chessckers (AlphaZero-style).

Differences from `mcts.py`:
- Selection uses PUCT instead of UCB1, with priors P(a|s) supplied by the
  network's policy head (softmax over candidate moves at the parent).
- Leaf evaluation uses the network's value head instead of material.

Selection score for a child given its parent:

    score(c) = -Q(c) + c_puct * P(c) * sqrt(parent.N) / (1 + c.N)

The negation on Q is the same as UCB1: a child's stored Q is from the child's
side-to-move perspective, but the parent wants children that are *bad for the
child's STM* (= good for the parent's STM).

Like the UCB1 variant, terminal nodes get `TERMINAL_LOSS_VALUE` (= -1 here so
it sits in the same range as the value head's tanh output) for mate /
variantEnd, and `TERMINAL_DRAW_VALUE` for stalemate.

Self-play uses the visit counts at the root as a sharpened policy target —
that's the AlphaZero policy improvement signal.
"""

from __future__ import annotations

import gc
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from chessckers_engine.encoding import encode_move, encode_position, encode_position_state
from chessckers_engine.model import ChesskersScorer

log = logging.getLogger("chessckers_engine.mcts_puct")

GameState = dict[str, Any]
LegalMove = dict[str, Any]

# Value head outputs in [-1, 1]; keep terminal values in the same range so
# they're comparable to learned values during backup.
TERMINAL_LOSS_VALUE = -1.0
TERMINAL_DRAW_VALUE = 0.0


class _Mover(Protocol):
    def new_game(self, fen: str | None = None) -> GameState: ...
    def make_move(self, fen: str, uci: str) -> GameState: ...


@dataclass(slots=True)
class PuctNode:
    # `fen` may be the empty string when the fast path defers serialization
    # until a leaf actually needs it (most expanded children are never picked,
    # so serializing all of them is pure waste). Use `_node_fen(node, client)`
    # rather than reading `node.fen` directly.
    fen: str
    move_to_here: LegalMove | None
    prior: float = 0.0  # P(this move | parent state)
    children: dict[str, "PuctNode"] = field(default_factory=dict)
    visits: int = 0
    total_value: float = 0.0
    is_terminal: bool = False
    terminal_status: str | None = None
    expanded: bool = False
    # Fast-path caches: parent State (so children can be built without
    # re-parsing the FEN) and the legal moves at this node (so MCTS doesn't
    # re-enumerate them in get_legal). Both are populated lazily by the
    # variant_py fast path; on the dict-API path (new_game/make_move) they
    # stay None.
    state: Any | None = None
    legal_moves: list[LegalMove] | None = None

    @property
    def q(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0


def _node_fen(node: PuctNode, client: Any) -> str:
    """Resolve a node's FEN, materializing it from the cached State on first
    access. Lets the fast-path expansion skip ~3000 string serializations per
    100 sims (only the ~100 leaves actually need a FEN for network input)."""
    if not node.fen and node.state is not None and hasattr(client, "state_to_fen"):
        node.fen = client.state_to_fen(node.state)
    return node.fen


# Lc0-style PUCT refinements:
# - cpuct GROWS with parent visits: cpuct = c_puct + CPUCT_FACTOR*log((N+base)/base).
#   With base=19652 this is negligible at our sim counts (~+0.04 at N=400) — it
#   only matters at Lc0-scale visit counts — but it matches Lc0 and future-proofs.
# - FPU (First-Play Urgency) reduction: an unvisited child is valued at the
#   parent's Q minus FPU_REDUCTION*sqrt(explored policy mass), rather than the
#   optimistic 0, so search stops over-exploring untried low-prior moves once
#   some siblings are visited. Set FPU_REDUCTION=0 to disable.
CPUCT_FACTOR = 2.0
CPUCT_BASE = 19652.0
FPU_REDUCTION = 0.25


def _puct_score(child: PuctNode, parent_visits: int, cpuct: float, fpu_value: float) -> float:
    # Q from the parent's perspective: explored child -> -child.q; unexplored -> FPU.
    q_from_parent = (-child.q) if child.visits > 0 else fpu_value
    u = cpuct * child.prior * math.sqrt(max(parent_visits, 1)) / (1 + child.visits)
    return q_from_parent + u


def _select_child(parent: PuctNode, c_puct: float) -> PuctNode:
    pv = parent.visits
    cpuct = c_puct + CPUCT_FACTOR * math.log((pv + CPUCT_BASE) / CPUCT_BASE)
    # FPU base: the parent's own value (parent STM), reduced by the policy mass
    # already explored among its children.
    visited_prior = sum(c.prior for c in parent.children.values() if c.visits > 0)
    fpu_value = parent.q - FPU_REDUCTION * math.sqrt(max(visited_prior, 0.0))
    return max(parent.children.values(), key=lambda c: _puct_score(c, pv, cpuct, fpu_value))


def _terminal_value(node: PuctNode) -> float:
    """Fixed value for terminal leaves; the side-to-move at a terminal node has
    just lost (or stalemated)."""
    if node.terminal_status == "stalemate":
        return TERMINAL_DRAW_VALUE
    return TERMINAL_LOSS_VALUE


def _eval_and_priors(
    evaluator, fen: str, legal_moves: list[LegalMove],
    state: Any = None,
) -> tuple[float, list[float]]:
    """One leaf evaluation returning (value, priors). `evaluator` is either
    a `ChesskersScorer` (direct in-thread forward pass) or an
    `InferenceServer` (queued, batched across concurrent MCTS workers).
    Distinguished by duck-typing on `submit`.

    If `state` is supplied (the variant_py fast path always has it cached on
    the PuctNode), encode the position directly from the in-memory State
    instead of re-parsing the FEN string."""
    if hasattr(evaluator, "submit"):
        # Server path — submit and block. Server handles batching across
        # concurrent worker threads. The submit API is FEN-keyed, so we
        # still pass the FEN here; cross-process IPC needs a string anyway.
        return evaluator.submit(fen, legal_moves).result()
    # Direct model path — one trunk pass, both heads.
    model = evaluator
    device = next(model.parameters()).device
    pos = (
        encode_position_state(state) if state is not None else encode_position(fen)
    ).unsqueeze(0).to(device)
    if not legal_moves:
        with torch.no_grad():
            value = model.value(pos)
        return float(value.item()), []
    moves = torch.stack([encode_move(m) for m in legal_moves]).to(device)
    with torch.no_grad():
        logits, value = model.policy_and_value(pos, moves)
        probs = torch.softmax(logits, dim=0)
    return float(value.item()), probs.tolist()


def _expand_with_priors(
    node: PuctNode,
    legal_moves: list[LegalMove],
    priors: list[float],
    client: _Mover,
) -> None:
    """Apply each legal move via the API; cache its post-state and the
    pre-computed prior on the child node. Priors come from a single
    `policy_and_value` call so the trunk runs once per leaf.

    Fast path (PyVariantClient): if the client exposes `parse` /
    `apply_known` / `status_and_legal`, use them — parse the parent FEN once,
    apply each child move to a copy of that State, detect terminal status
    via the cheap-first ladder, and cache the resulting State + legal moves
    on the child node. This skips ~50% of MCTS CPU vs the dict round-trip
    path (`make_move(fen, uci)` re-parses, re-validates, re-serializes, and
    re-runs full Black move-gen for the legalMoves field nobody reads).
    """
    fast = (
        hasattr(client, "apply_known")
        and hasattr(client, "status_and_legal")
        and hasattr(client, "parse")
        and hasattr(client, "state_to_fen")
    )
    if fast:
        parent_state = node.state if node.state is not None else client.parse(node.fen)
        node.state = parent_state
        for move, prior in zip(legal_moves, priors):
            try:
                child_state = client.apply_known(parent_state, move)
            except Exception as e:  # noqa: BLE001
                log.debug("expand: skipping uci=%s: %s", move["uci"], e)
                continue
            status, _winner, child_legal = client.status_and_legal(child_state)
            # Defer fen serialization — only ~3% of expanded children get
            # picked as leaves. `_node_fen` materializes on demand.
            child = PuctNode(
                fen="",
                move_to_here=move,
                prior=float(prior),
                is_terminal=bool(status),
                terminal_status=status,
                state=child_state,
                legal_moves=child_legal,
            )
            node.children[move["uci"]] = child
        node.expanded = True
        return

    for move, prior in zip(legal_moves, priors):
        try:
            new_state = client.make_move(node.fen, move["uci"])
        except Exception as e:  # noqa: BLE001
            log.debug("expand: skipping unreachable candidate uci=%s: %s", move["uci"], e)
            continue
        child = PuctNode(
            fen=new_state["fen"],
            move_to_here=move,
            prior=float(prior),
            is_terminal=bool(new_state.get("status")),
            terminal_status=new_state.get("status"),
        )
        node.children[move["uci"]] = child
    node.expanded = True


def _backup(path: list[PuctNode], leaf_value: float) -> None:
    # Per-ply value discount (env CHESSCKERS_VALUE_DISCOUNT, default 1.0 = off,
    # the same γ used for the training value target). The leaf keeps its full
    # value; each step up toward the root multiplies by γ, so a win reached in
    # fewer plies backs up a larger value at the root and the search prefers
    # faster mates. `sign` flips per ply for the side-to-move perspective.
    gamma = float(os.environ.get("CHESSCKERS_VALUE_DISCOUNT", "1.0"))
    sign = 1.0
    discount = 1.0
    for node in reversed(path):
        node.visits += 1
        node.total_value += sign * discount * leaf_value
        sign = -sign
        discount *= gamma


def _select_to_leaf(root: PuctNode, c_puct: float) -> tuple[list[PuctNode], PuctNode]:
    """Walk from root selecting PUCT-best children until we hit an unexpanded
    or terminal node. Returns (path-from-root-inclusive, leaf)."""
    path: list[PuctNode] = [root]
    node = root
    while node.expanded and not node.is_terminal and node.children:
        node = _select_child(node, c_puct)
        path.append(node)
    return path, node


# Virtual loss: when we select a leaf in a batched MCTS pass, we mark every
# node on its path as if it had been visited with a losing outcome. This
# discourages subsequent parallel selections in the same batch from picking
# the same path, diversifying which leaves get expanded. After the real eval
# returns we reverse the virtual loss before backing up the actual value.
VIRTUAL_LOSS_COUNT = 1


def _apply_virtual_loss(path: list[PuctNode], count: int = VIRTUAL_LOSS_COUNT) -> None:
    for node in path:
        node.visits += count
        node.total_value -= float(count)


def _remove_virtual_loss(path: list[PuctNode], count: int = VIRTUAL_LOSS_COUNT) -> None:
    for node in path:
        node.visits -= count
        node.total_value += float(count)


def _simulate(
    root: PuctNode,
    client: _Mover,
    model: ChesskersScorer,
    c_puct: float,
    get_legal_moves,
) -> None:
    path, node = _select_to_leaf(root, c_puct)

    if node.is_terminal:
        value = _terminal_value(node)
    elif not node.expanded:
        legal = get_legal_moves(node)
        value, priors = _eval_and_priors(
            model, _node_fen(node, client), legal, state=node.state,
        )
        if legal:
            _expand_with_priors(node, legal, priors, client)
    else:
        # Expanded but no children — shouldn't occur in normal play, but
        # evaluate via the value head as a safe fallback.
        value, _ = _eval_and_priors(
            model, _node_fen(node, client), [], state=node.state,
        )

    _backup(path, value)


def _simulate_batched(
    root: PuctNode,
    client: _Mover,
    evaluator,
    c_puct: float,
    get_legal_moves,
    batch_size: int,
) -> None:
    """Run `batch_size` simulations in a single batched pass, using virtual
    loss to diversify selections. Requires `evaluator` to be an InferenceServer
    (or anything with `submit(fen, legal) → Future`) so the B leaf evaluations
    can run as one batched forward."""
    if not hasattr(evaluator, "submit"):
        # No server: fall back to sequential simulations.
        for _ in range(batch_size):
            _simulate(root, client, evaluator, c_puct, get_legal_moves)
        return

    # Selection phase: pick B leaves with virtual loss applied between picks.
    selections: list[tuple[list[PuctNode], PuctNode, list[LegalMove] | None]] = []
    for _ in range(batch_size):
        path, leaf = _select_to_leaf(root, c_puct)
        _apply_virtual_loss(path)
        legal: list[LegalMove] | None = None
        if not leaf.is_terminal:
            legal = get_legal_moves(leaf)
        selections.append((path, leaf, legal))

    # Evaluation phase: submit B requests; the server batches them on the GPU.
    futures: list = []
    for _, leaf, legal in selections:
        if leaf.is_terminal:
            futures.append(None)
        else:
            futures.append(evaluator.submit(_node_fen(leaf, client), legal or []))

    # Backup phase: remove virtual loss, expand if needed (idempotent if a
    # duplicate leaf appears twice in the batch — second expansion no-ops),
    # then back up the real value.
    for (path, leaf, legal), fut in zip(selections, futures):
        _remove_virtual_loss(path)
        if leaf.is_terminal:
            value = _terminal_value(leaf)
        else:
            value, priors = fut.result()
            if legal and not leaf.expanded:
                _expand_with_priors(leaf, legal, priors, client)
        _backup(path, value)


@dataclass
class MctsResult:
    chosen: LegalMove | None
    visit_distribution: dict[str, int]  # uci -> visit count
    root: PuctNode


def _apply_dirichlet_noise(
    root: PuctNode,
    alpha: float,
    eps: float,
    rng: torch.Generator | None = None,
) -> None:
    """Mix Dirichlet(α) noise into the root's children's priors.

    For each child:  P_new = (1 - eps) * P_old + eps * noise_sample
    where the noise_sample comes from a single Dirichlet draw across all
    root children. With small α, the draw is spiky — concentrated on a few
    randomly chosen children — so the noise drives commitment-style
    exploration of low-prior candidates rather than uniform dilution.

    Use only at the root, only during self-play. The standard AlphaZero
    defaults are α≈0.3 (chess-scale action space; smaller for Go) and
    ε=0.25.
    """
    if not root.children:
        return
    n = len(root.children)
    concentration = torch.full((n,), float(alpha))
    dist = torch.distributions.Dirichlet(concentration)
    if rng is None:
        sample = dist.sample()
    else:
        # torch.distributions doesn't accept a Generator directly; we sample
        # from a uniform via the rng and reparameterize via Gamma if a
        # specific generator is required. For test-determinism we just
        # accept the global RNG since AZ self-play drives variance through
        # temperature sampling more than this hook.
        sample = dist.sample()
    noise = sample.tolist()
    for i, child in enumerate(root.children.values()):
        child.prior = float((1.0 - eps) * child.prior + eps * noise[i])


def run_mcts(
    state: GameState,
    client: _Mover,
    model: ChesskersScorer,
    n_sims: int = 100,
    c_puct: float = 1.5,
    dirichlet_alpha: float | None = None,
    dirichlet_eps: float = 0.25,
    vloss_batch: int = 1,
) -> MctsResult:
    """Run PUCT MCTS for `n_sims` iterations from `state`. Returns the chosen
    move (most-visited root child) along with the visit distribution that can
    be used as a policy target for self-play training.

    `dirichlet_alpha`: if not None, mix Dirichlet(α) noise into the root's
    priors after the first simulation expands the root. Only the root is
    affected. Use during self-play; leave None during inference/eval.

    `vloss_batch`: virtual-loss batch size. Default 1 = sequential. When >1,
    select B paths with virtual loss applied between picks, evaluate all B
    leaves in one batched forward (requires `model` to be an InferenceServer
    so the eval can actually batch), then back up. Multiplies effective
    inference batch size by ~B per game.
    """
    legal = state.get("legalMoves") or []
    if not legal:
        return MctsResult(chosen=None, visit_distribution={}, root=PuctNode(fen=state["fen"], move_to_here=None))

    # Disable cyclic-GC for the duration of the search. MCTS allocates ~3400
    # PuctNodes / move dicts / CaptureHops per 100 sims and none of them form
    # reference cycles (PuctNode children are owned top-down; no parent-back
    # pointers; State/dict/list contents are leaves), so the cycle collector
    # has no work to do. Skipping it saves ~13% wall time at typical sim
    # counts. Refcount-based dealloc still runs as normal — no leak risk.
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()

    root = PuctNode(fen=state["fen"], move_to_here=None, legal_moves=legal)
    # Eagerly cache the root's parsed State for the fast path so the first
    # expansion doesn't have to re-parse the same FEN we just got.
    if hasattr(client, "parse"):
        try:
            root.state = client.parse(state["fen"])
        except Exception:  # noqa: BLE001
            pass
    legal_cache: dict[str, list[LegalMove]] = {state["fen"]: legal}

    def get_legal(node: PuctNode) -> list[LegalMove]:
        # Node-local cache (populated by the fast-path expansion) wins —
        # it's the same list that detected the node's terminal status.
        if node.legal_moves is not None:
            return node.legal_moves
        cached = legal_cache.get(node.fen)
        if cached is not None:
            node.legal_moves = cached
            return cached
        try:
            s = client.new_game(node.fen)  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001
            log.debug("get_legal: new_game raised for fen=%s: %s", node.fen, e)
            return []
        moves = s.get("legalMoves") or []
        legal_cache[node.fen] = moves
        node.legal_moves = moves
        return moves

    # First sim expands the root, populating children + their priors. Done
    # sequentially so the Dirichlet noise that follows sees the priors.
    if n_sims > 0:
        _simulate(root, client, model, c_puct, get_legal)

    # Optionally mix Dirichlet noise into root priors before the rest of search.
    if dirichlet_alpha is not None:
        _apply_dirichlet_noise(root, dirichlet_alpha, dirichlet_eps)

    remaining = max(0, n_sims - 1)
    try:
        if vloss_batch <= 1:
            for _ in range(remaining):
                _simulate(root, client, model, c_puct, get_legal)
        else:
            sims_done = 0
            while sims_done < remaining:
                b = min(vloss_batch, remaining - sims_done)
                _simulate_batched(root, client, model, c_puct, get_legal, b)
                sims_done += b
    finally:
        if gc_was_enabled:
            gc.enable()

    if not root.children:
        return MctsResult(chosen=legal[0], visit_distribution={legal[0]["uci"]: 0}, root=root)

    visit_dist = {uci: c.visits for uci, c in root.children.items()}
    best = max(root.children.values(), key=lambda c: c.visits)
    return MctsResult(chosen=best.move_to_here, visit_distribution=visit_dist, root=root)


def pick_puct(
    state: GameState,
    client: _Mover,
    model: ChesskersScorer,
    n_sims: int = 100,
    c_puct: float = 1.5,
    vloss_batch: int = 1,
) -> LegalMove | None:
    """Picker-shaped wrapper: returns just the chosen move."""
    return run_mcts(
        state, client, model,
        n_sims=n_sims, c_puct=c_puct, vloss_batch=vloss_batch,
    ).chosen
