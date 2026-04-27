"""Heuristic MCTS for Chessckers.

Builds a search tree rooted at the current position. Each iteration:
  1. Select a leaf via UCB1 (descending the tree from the root).
  2. If the leaf is non-terminal, expand it by querying the API for the
     legal moves and the post-move FEN of each.
  3. Evaluate the leaf (or its expanded children's parent — see below)
     with `material_for_side_to_move(fen, king_value=1000)`.
  4. Back up the value along the path, alternating signs (zero-sum game).

After `n_sims` iterations, the most-visited child of the root is returned
as the chosen move. This is the standard "robust child" selection that
AlphaZero papers and most modern MCTS implementations use.

Design notes:
- Values are stored from each node's side-to-move perspective. Backup
  alternates sign so that a leaf evaluated at +V contributes +V to the
  leaf's STM and -V to the leaf's parent (whose STM is the opposite).
- Selection from a parent picks the child that is *worst* for the child's
  STM, equivalently best for the parent. So the score function is
  `-child.Q + c_puct * sqrt(ln(parent.visits) / child.visits)`.
- Terminal nodes are detected via the `status` field returned by the
  API. Their value is fixed: a large negative for mate/variantEnd
  (the STM is always the loser at a terminal node — they couldn't move),
  zero for stalemate.
- Server inconsistencies (the e.p. bug etc.) are skipped with a debug
  log; if every candidate at a node fails, the node has no children and
  its value comes from material at expansion time.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from chessckers_engine.material import material_for_side_to_move

log = logging.getLogger("chessckers_engine.mcts")

GameState = dict[str, Any]
LegalMove = dict[str, Any]

TERMINAL_LOSS_VALUE = -10000.0  # dominates any material delta
TERMINAL_DRAW_VALUE = 0.0


class _Mover(Protocol):
    def make_move(self, fen: str, uci: str) -> GameState: ...


@dataclass
class Node:
    fen: str
    move_to_here: LegalMove | None  # None for the root
    children: dict[str, "Node"] = field(default_factory=dict)
    visits: int = 0
    total_value: float = 0.0
    is_terminal: bool = False
    terminal_status: str | None = None
    expanded: bool = False  # children have been populated

    @property
    def q(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0


def _ucb1_score(child: Node, parent_visits: int, c_puct: float) -> float:
    if child.visits == 0:
        return math.inf
    log_parent = math.log(max(parent_visits, 1))
    q_from_parent = -child.q
    u = c_puct * math.sqrt(log_parent / child.visits)
    return q_from_parent + u


def _select_child(parent: Node, c_puct: float) -> Node:
    return max(parent.children.values(), key=lambda c: _ucb1_score(c, parent.visits, c_puct))


def _expand(node: Node, legal_moves: list[LegalMove], client: _Mover) -> None:
    """Populate `node.children` by applying each legal move via the API.

    Skips candidates whose make_move call raises (scalachess server bug
    around e.p.-shaped captures against Chessckers towers).
    """
    for move in legal_moves:
        try:
            new_state = client.make_move(node.fen, move["uci"])
        except Exception as e:  # noqa: BLE001
            log.debug("expand: skipping unreachable candidate uci=%s: %s", move["uci"], e)
            continue
        child = Node(
            fen=new_state["fen"],
            move_to_here=move,
            is_terminal=bool(new_state.get("status")),
            terminal_status=new_state.get("status"),
        )
        node.children[move["uci"]] = child
    node.expanded = True


def _evaluate(node: Node) -> float:
    if node.is_terminal:
        if node.terminal_status == "stalemate":
            return TERMINAL_DRAW_VALUE
        return TERMINAL_LOSS_VALUE
    return float(material_for_side_to_move(node.fen, king_value=1000))


def _backup(path: list[Node], leaf_value: float) -> None:
    sign = 1.0
    for node in reversed(path):
        node.visits += 1
        node.total_value += sign * leaf_value
        sign = -sign


def _simulate(root: Node, client: _Mover, c_puct: float, get_legal_moves) -> None:
    """One MCTS iteration. `get_legal_moves(node) -> list[LegalMove]` is supplied
    by the caller so the root can use the legalMoves it already has and avoid an
    extra round-trip; expanded children fetch their own."""
    path: list[Node] = [root]
    node = root
    # Selection: descend until we hit an unexpanded or terminal node.
    while node.expanded and not node.is_terminal and node.children:
        node = _select_child(node, c_puct)
        path.append(node)

    # Expansion: if we landed on an unexpanded non-terminal, expand it.
    if not node.is_terminal and not node.expanded:
        legal = get_legal_moves(node)
        if legal:
            _expand(node, legal, client)

    # Evaluation: heuristic value of the leaf itself.
    value = _evaluate(node)
    _backup(path, value)


def pick_mcts(
    state: GameState,
    client: _Mover,
    n_sims: int = 100,
    c_puct: float = 1.4,
) -> LegalMove | None:
    """Run MCTS for `n_sims` iterations starting from `state`. Returns the
    most-visited root child's move, or None if there are no legal moves."""
    legal = state.get("legalMoves") or []
    if not legal:
        return None

    root = Node(fen=state["fen"], move_to_here=None)

    # Root's legalMoves come from the caller (no extra round-trip).
    # Children must fetch their own. We do that by calling client.new_game(fen)
    # — which the API treats as "give me the GameState for this FEN" — and
    # caching the result.
    legal_cache: dict[str, list[LegalMove]] = {state["fen"]: legal}

    def get_legal(node: Node) -> list[LegalMove]:
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

    for _ in range(n_sims):
        _simulate(root, client, c_puct, get_legal)

    if not root.children:
        # Expansion failed for every candidate (every move was rejected by the
        # server). Fall back to the first legal move so the caller never hangs.
        return legal[0]
    best = max(root.children.values(), key=lambda c: c.visits)
    return best.move_to_here
