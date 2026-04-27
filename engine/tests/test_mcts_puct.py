from typing import Any

import torch

from chessckers_engine.mcts_puct import (
    TERMINAL_DRAW_VALUE,
    TERMINAL_LOSS_VALUE,
    PuctNode,
    _backup,
    _evaluate_with_model,
    _expand,
    _puct_score,
    pick_puct,
    run_mcts,
)
from chessckers_engine.model import ChesskersScorer


FEN_W = "8/8/8/8/8/8/8/4K3 w - - 0 1"
FEN_B = "8/8/8/8/8/8/8/4K3 b - - 0 1"


def _move(uci: str) -> dict:
    return {"uci": uci, "from": "a1", "to": "a2"}


# ---- PUCT score formula ----


def test_puct_score_unvisited_child_uses_only_prior_term():
    """Q is 0 (visits=0), exploration term = c_puct * prior * sqrt(parent_N) / 1."""
    child = PuctNode(fen=FEN_W, move_to_here=None, prior=0.5, visits=0)
    parent_visits = 4
    c_puct = 2.0
    expected = -0.0 + c_puct * 0.5 * (parent_visits ** 0.5) / (1 + 0)
    assert abs(_puct_score(child, parent_visits, c_puct) - expected) < 1e-9


def test_puct_score_higher_prior_wins_when_q_and_visits_equal():
    high = PuctNode(fen=FEN_W, move_to_here=None, prior=0.8, visits=2, total_value=0.4)
    low = PuctNode(fen=FEN_W, move_to_here=None, prior=0.1, visits=2, total_value=0.4)
    s_high = _puct_score(high, parent_visits=10, c_puct=1.5)
    s_low = _puct_score(low, parent_visits=10, c_puct=1.5)
    assert s_high > s_low


def test_puct_score_visited_child_q_dominates_eventually():
    """A frequently visited losing child (low Q) loses to an unvisited child."""
    visited_loser = PuctNode(fen=FEN_W, move_to_here=None, prior=0.5, visits=100, total_value=-90.0)
    unvisited = PuctNode(fen=FEN_W, move_to_here=None, prior=0.05, visits=0, total_value=0.0)
    s_loser = _puct_score(visited_loser, parent_visits=200, c_puct=1.5)
    s_unvisited = _puct_score(unvisited, parent_visits=200, c_puct=1.5)
    assert s_unvisited > s_loser


# ---- Evaluation ----


def test_evaluate_uses_value_head_for_non_terminal_leaves():
    torch.manual_seed(0)
    model = ChesskersScorer().eval()
    node = PuctNode(fen=FEN_W, move_to_here=None, is_terminal=False)
    v = _evaluate_with_model(node, model)
    assert -1.0 <= v <= 1.0


def test_evaluate_returns_terminal_loss_value_for_mate():
    model = ChesskersScorer().eval()
    node = PuctNode(fen=FEN_W, move_to_here=None, is_terminal=True, terminal_status="mate")
    assert _evaluate_with_model(node, model) == TERMINAL_LOSS_VALUE


def test_evaluate_returns_terminal_loss_value_for_variant_end():
    model = ChesskersScorer().eval()
    node = PuctNode(fen=FEN_W, move_to_here=None, is_terminal=True, terminal_status="variantEnd")
    assert _evaluate_with_model(node, model) == TERMINAL_LOSS_VALUE


def test_evaluate_returns_zero_for_stalemate():
    model = ChesskersScorer().eval()
    node = PuctNode(fen=FEN_W, move_to_here=None, is_terminal=True, terminal_status="stalemate")
    assert _evaluate_with_model(node, model) == TERMINAL_DRAW_VALUE


# ---- Backup ----


def test_backup_alternates_sign_along_path():
    a = PuctNode(fen=FEN_W, move_to_here=None)
    b = PuctNode(fen=FEN_B, move_to_here=None)
    c = PuctNode(fen=FEN_W, move_to_here=None)
    _backup([a, b, c], leaf_value=0.6)
    assert (a.visits, a.total_value) == (1, 0.6)
    assert (b.visits, b.total_value) == (1, -0.6)
    assert (c.visits, c.total_value) == (1, 0.6)


# ---- Expansion + run_mcts (stub-driven) ----


class _StubClient:
    def __init__(self, table: dict[tuple[str, str], dict[str, Any]]) -> None:
        self.table = table

    def new_game(self, fen=None):
        return {"fen": fen, "legalMoves": []}

    def make_move(self, fen, uci):
        if (fen, uci) not in self.table:
            raise RuntimeError("rejected")
        return self.table[(fen, uci)]


def test_expand_skips_candidates_that_raise():
    parent = PuctNode(fen=FEN_W, move_to_here=None)
    moves = [_move("GOOD"), _move("BAD")]
    client = _StubClient({(FEN_W, "GOOD"): {"fen": FEN_B, "legalMoves": []}})
    model = ChesskersScorer().eval()
    _expand(parent, moves, client, model)
    assert "GOOD" in parent.children and "BAD" not in parent.children
    # Each child has a prior; they sum to 1 across all legal moves before filtering.
    # Surviving child kept its assigned prior (>= 0).
    assert 0.0 <= parent.children["GOOD"].prior <= 1.0


def test_run_mcts_returns_singleton_when_only_one_legal_move():
    only = _move("ONLY")
    state = {"fen": FEN_W, "legalMoves": [only]}
    client = _StubClient({(FEN_W, "ONLY"): {"fen": FEN_B, "legalMoves": []}})
    model = ChesskersScorer().eval()
    result = run_mcts(state, client, model, n_sims=5)
    assert result.chosen is only
    assert result.visit_distribution == {"ONLY": result.root.children["ONLY"].visits}


def test_run_mcts_prefers_winning_terminal_branch():
    """One move leads to variantEnd (we win — terminal value = -1 from
    opponent's perspective, which is +1 for us via backup). The other leads
    to stalemate (draw, value = 0)."""
    win = _move("WIN")
    draw = _move("DRAW")
    state = {"fen": FEN_W, "legalMoves": [win, draw]}
    client = _StubClient(
        {
            (FEN_W, "WIN"): {"fen": FEN_B, "status": "variantEnd"},
            (FEN_W, "DRAW"): {"fen": FEN_B, "status": "stalemate"},
        }
    )
    model = ChesskersScorer().eval()
    result = run_mcts(state, client, model, n_sims=30)
    assert result.chosen is win


def test_run_mcts_returns_visit_distribution_summing_to_n_sims():
    """The visit counts of root's children sum to one per simulation that
    descended into them. Since every simulation either expands the root (root
    visits += 1) or descends from root (also bumps a child by 1), in steady
    state child visits + 1 (the initial root visit during root expansion) ≈ n_sims."""
    a, b, c = _move("A"), _move("B"), _move("C")
    state = {"fen": FEN_W, "legalMoves": [a, b, c]}
    client = _StubClient(
        {
            (FEN_W, "A"): {"fen": FEN_B, "legalMoves": []},
            (FEN_W, "B"): {"fen": FEN_B, "legalMoves": []},
            (FEN_W, "C"): {"fen": FEN_B, "legalMoves": []},
        }
    )
    model = ChesskersScorer().eval()
    result = run_mcts(state, client, model, n_sims=12)
    total_child_visits = sum(result.visit_distribution.values())
    # First sim expands root and visits root only (no children visited yet).
    # Subsequent sims descend into a child each. So total_child_visits = n_sims - 1.
    assert total_child_visits == 11


def test_pick_puct_falls_back_when_every_candidate_fails():
    a, b = _move("A"), _move("B")
    state = {"fen": FEN_W, "legalMoves": [a, b]}
    client = _StubClient({})  # every make_move raises
    model = ChesskersScorer().eval()
    chosen = pick_puct(state, client, model, n_sims=5)
    assert chosen is a
