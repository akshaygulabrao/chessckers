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

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from chessckers_engine.encoding import encode_move, encode_position
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


@dataclass
class PuctNode:
    fen: str
    move_to_here: LegalMove | None
    prior: float = 0.0  # P(this move | parent state)
    children: dict[str, "PuctNode"] = field(default_factory=dict)
    visits: int = 0
    total_value: float = 0.0
    is_terminal: bool = False
    terminal_status: str | None = None
    expanded: bool = False

    @property
    def q(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0


def _puct_score(child: PuctNode, parent_visits: int, c_puct: float) -> float:
    q_from_parent = -child.q
    u = c_puct * child.prior * math.sqrt(max(parent_visits, 1)) / (1 + child.visits)
    return q_from_parent + u


def _select_child(parent: PuctNode, c_puct: float) -> PuctNode:
    return max(parent.children.values(), key=lambda c: _puct_score(c, parent.visits, c_puct))


def _evaluate_with_model(node: PuctNode, model: ChesskersScorer) -> float:
    """Use the value head for non-terminal leaves; fixed scalars for terminals."""
    if node.is_terminal:
        if node.terminal_status == "stalemate":
            return TERMINAL_DRAW_VALUE
        return TERMINAL_LOSS_VALUE
    pos = encode_position(node.fen).unsqueeze(0)
    with torch.no_grad():
        v = model.value(pos)
    return float(v.item())


def _compute_priors(
    model: ChesskersScorer, fen: str, legal_moves: list[LegalMove]
) -> list[float]:
    """Softmax over the policy head's logits at this position. Returns a list
    of probabilities aligned with `legal_moves` (sums to 1, length = len(legal_moves))."""
    if not legal_moves:
        return []
    pos = encode_position(fen).unsqueeze(0)
    moves = torch.stack([encode_move(m) for m in legal_moves])
    with torch.no_grad():
        logits = model(pos, moves)
        probs = torch.softmax(logits, dim=0)
    return probs.tolist()


def _expand(
    node: PuctNode,
    legal_moves: list[LegalMove],
    client: _Mover,
    model: ChesskersScorer,
) -> None:
    """Apply each legal move via the API; cache its post-state and prior on the child node."""
    priors = _compute_priors(model, node.fen, legal_moves)
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
    sign = 1.0
    for node in reversed(path):
        node.visits += 1
        node.total_value += sign * leaf_value
        sign = -sign


def _simulate(
    root: PuctNode,
    client: _Mover,
    model: ChesskersScorer,
    c_puct: float,
    get_legal_moves,
) -> None:
    path: list[PuctNode] = [root]
    node = root
    while node.expanded and not node.is_terminal and node.children:
        node = _select_child(node, c_puct)
        path.append(node)

    if not node.is_terminal and not node.expanded:
        legal = get_legal_moves(node)
        if legal:
            _expand(node, legal, client, model)

    value = _evaluate_with_model(node, model)
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
) -> MctsResult:
    """Run PUCT MCTS for `n_sims` iterations from `state`. Returns the chosen
    move (most-visited root child) along with the visit distribution that can
    be used as a policy target for self-play training.

    `dirichlet_alpha`: if not None, mix Dirichlet(α) noise into the root's
    priors after the first simulation expands the root. Only the root is
    affected. Use during self-play; leave None during inference/eval.
    """
    legal = state.get("legalMoves") or []
    if not legal:
        return MctsResult(chosen=None, visit_distribution={}, root=PuctNode(fen=state["fen"], move_to_here=None))

    root = PuctNode(fen=state["fen"], move_to_here=None)
    legal_cache: dict[str, list[LegalMove]] = {state["fen"]: legal}

    def get_legal(node: PuctNode) -> list[LegalMove]:
        cached = legal_cache.get(node.fen)
        if cached is not None:
            return cached
        try:
            s = client.new_game(node.fen)  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001
            log.debug("get_legal: new_game raised for fen=%s: %s", node.fen, e)
            return []
        moves = s.get("legalMoves") or []
        legal_cache[node.fen] = moves
        return moves

    # First sim expands the root, populating children + their priors.
    if n_sims > 0:
        _simulate(root, client, model, c_puct, get_legal)

    # Optionally mix Dirichlet noise into root priors before the rest of search.
    if dirichlet_alpha is not None:
        _apply_dirichlet_noise(root, dirichlet_alpha, dirichlet_eps)

    for _ in range(max(0, n_sims - 1)):
        _simulate(root, client, model, c_puct, get_legal)

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
) -> LegalMove | None:
    """Picker-shaped wrapper: returns just the chosen move."""
    return run_mcts(state, client, model, n_sims=n_sims, c_puct=c_puct).chosen
