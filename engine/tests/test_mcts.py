from typing import Any

import pytest

from chessckers_engine.mcts import (
    TERMINAL_DRAW_VALUE,
    TERMINAL_LOSS_VALUE,
    Node,
    _backup,
    _evaluate,
    _expand,
    _ucb1_score,
    pick_mcts,
)


# Minimal valid FEN (any STM, any board) — only the parser cares about shape.
FEN_W = "8/8/8/8/8/8/8/4K3 w - - 0 1"
FEN_B = "8/8/8/8/8/8/8/4K3 b - - 0 1"
FEN_BLACK_HAS_STONE_W = "8/8/8/8/8/8/8/4K3[d4:s] w - - 0 1"


def test_ucb1_unvisited_child_returns_infinity():
    child = Node(fen=FEN_W, move_to_here=None, visits=0, total_value=0.0)
    assert _ucb1_score(child, parent_visits=10, c_puct=1.4) == float("inf")


def test_ucb1_visited_child_returns_finite():
    child = Node(fen=FEN_W, move_to_here=None, visits=5, total_value=10.0)
    score = _ucb1_score(child, parent_visits=20, c_puct=1.4)
    assert score < float("inf") and score == score  # finite, not NaN


def test_evaluate_returns_material_for_non_terminal_leaf():
    node = Node(fen=FEN_BLACK_HAS_STONE_W, move_to_here=None, is_terminal=False)
    # White material - Black material from white-to-move perspective:
    # White has just K (1000), Black has 1 stone (1). raw = 999. STM=white, no flip.
    assert _evaluate(node) == 999.0


def test_evaluate_returns_loss_value_for_mate():
    node = Node(fen=FEN_W, move_to_here=None, is_terminal=True, terminal_status="mate")
    assert _evaluate(node) == TERMINAL_LOSS_VALUE


def test_evaluate_returns_loss_value_for_variant_end():
    node = Node(fen=FEN_W, move_to_here=None, is_terminal=True, terminal_status="variantEnd")
    assert _evaluate(node) == TERMINAL_LOSS_VALUE


def test_evaluate_returns_zero_for_stalemate():
    node = Node(fen=FEN_W, move_to_here=None, is_terminal=True, terminal_status="stalemate")
    assert _evaluate(node) == TERMINAL_DRAW_VALUE


def test_backup_alternates_sign_along_path():
    a = Node(fen=FEN_W, move_to_here=None)
    b = Node(fen=FEN_B, move_to_here=None)
    c = Node(fen=FEN_W, move_to_here=None)
    _backup([a, b, c], leaf_value=10.0)
    # path[2]=c is the leaf -> +10; b -> -10; a -> +10
    assert (a.visits, a.total_value) == (1, 10.0)
    assert (b.visits, b.total_value) == (1, -10.0)
    assert (c.visits, c.total_value) == (1, 10.0)


def test_backup_accumulates_over_calls():
    a = Node(fen=FEN_W, move_to_here=None)
    b = Node(fen=FEN_B, move_to_here=None)
    _backup([a, b], leaf_value=5.0)
    _backup([a, b], leaf_value=3.0)
    assert (a.visits, a.total_value) == (2, -8.0)  # leaf is b, parent a sees -leaf
    assert (b.visits, b.total_value) == (2, 8.0)


# ---- Expansion tests with a stub client ----


class _StubClient:
    """Maps (fen, uci) -> post-state dict; raises for missing keys."""

    def __init__(self, table: dict[tuple[str, str], dict[str, Any]]) -> None:
        self.table = table
        self.calls: list[tuple[str, str]] = []

    def new_game(self, fen: str | None = None) -> dict[str, Any]:
        return {"fen": fen, "legalMoves": []}

    def make_move(self, fen: str, uci: str) -> dict[str, Any]:
        self.calls.append((fen, uci))
        if (fen, uci) not in self.table:
            raise RuntimeError("server rejected")
        return self.table[(fen, uci)]


def test_expand_skips_candidates_that_raise():
    parent = Node(fen=FEN_W, move_to_here=None)
    moves = [{"uci": "GOOD", "from": "a", "to": "b"}, {"uci": "BAD"}]
    client = _StubClient({(FEN_W, "GOOD"): {"fen": FEN_B, "legalMoves": []}})
    _expand(parent, moves, client)
    assert "GOOD" in parent.children
    assert "BAD" not in parent.children
    assert parent.expanded is True


def test_expand_marks_children_terminal_when_status_present():
    parent = Node(fen=FEN_W, move_to_here=None)
    moves = [{"uci": "WIN"}]
    client = _StubClient({(FEN_W, "WIN"): {"fen": FEN_B, "status": "variantEnd"}})
    _expand(parent, moves, client)
    win_child = parent.children["WIN"]
    assert win_child.is_terminal
    assert win_child.terminal_status == "variantEnd"


# ---- Top-level pick_mcts smoke tests ----


def test_pick_mcts_returns_none_when_no_legal_moves():
    state = {"fen": FEN_W, "legalMoves": []}
    assert pick_mcts(state, _StubClient({}), n_sims=10) is None


def test_pick_mcts_returns_singleton_when_only_one_move():
    move = {"uci": "M1"}
    state = {"fen": FEN_W, "legalMoves": [move]}
    client = _StubClient({(FEN_W, "M1"): {"fen": FEN_B, "legalMoves": []}})
    assert pick_mcts(state, client, n_sims=5) is move


def test_pick_mcts_prefers_winning_terminal_over_losing_terminal():
    """Two moves at root: one leads to a stalemate (draw, value=0), the other
    to a position where it's now Black's turn and Black has lost (variantEnd).
    From White's perspective the variantEnd is a win, the stalemate is a draw,
    so MCTS should prefer the variantEnd."""
    win_move = {"uci": "WIN"}
    draw_move = {"uci": "DRAW"}
    state = {"fen": FEN_W, "legalMoves": [win_move, draw_move]}
    # WIN leads to a child whose status is variantEnd (White wins)
    # DRAW leads to a child whose status is stalemate (draw)
    client = _StubClient(
        {
            (FEN_W, "WIN"): {"fen": FEN_B, "status": "variantEnd"},
            (FEN_W, "DRAW"): {"fen": FEN_B, "status": "stalemate"},
        }
    )
    chosen = pick_mcts(state, client, n_sims=20)
    assert chosen is win_move


def test_pick_mcts_falls_back_to_first_legal_when_every_candidate_fails():
    """Server rejects every candidate at root → expand produces no children.
    pick_mcts must still return *something* (the first legal move)."""
    a = {"uci": "A"}
    b = {"uci": "B"}
    state = {"fen": FEN_W, "legalMoves": [a, b]}
    client = _StubClient({})  # empty table → every make_move raises
    chosen = pick_mcts(state, client, n_sims=5)
    assert chosen is a


def test_pick_mcts_picks_higher_material_terminal_branch():
    """One root move leads to a position where Black has lost a stone (good
    for White, who just moved). The other leads to a quiet position with no
    captures. With king_value=1000 dominating, the quiet branch should evaluate
    near zero net but the capture branch will show as a loss for the next-mover
    (Black), i.e. positive for White. MCTS should prefer it."""
    capture = {"uci": "TAKE"}
    quiet = {"uci": "QUIET"}
    # After "TAKE": Black has lost their only stone; black-to-move; white still has K.
    after_take = {"fen": "8/8/8/8/8/8/8/4K3 b - - 0 1", "legalMoves": []}
    # After "QUIET": same FEN but Black still has the stone.
    after_quiet = {"fen": FEN_BLACK_HAS_STONE_W.replace(" w ", " b "), "legalMoves": []}
    client = _StubClient(
        {
            (FEN_BLACK_HAS_STONE_W, "TAKE"): after_take,
            (FEN_BLACK_HAS_STONE_W, "QUIET"): after_quiet,
        }
    )
    state = {"fen": FEN_BLACK_HAS_STONE_W, "legalMoves": [capture, quiet]}
    chosen = pick_mcts(state, client, n_sims=20)
    assert chosen is capture
