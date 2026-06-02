"""Tree-reuse (Lc0-style) correctness for run_mcts / play_az_game.

The dangerous failure mode is searching a STALE subtree that doesn't match the
actual position — that would feed garbage policy/value targets into training.
These tests pin: reuse continues the CORRECT position, a mismatched subtree is
discarded (fresh search), a full game runs with only legal moves, and reuse
actually reduces NN work (so it's reusing carried visits, not always falling back).
"""
import os

import torch

from chessckers_engine.mcts_puct import _node_fen, run_mcts
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.selfplay_az import play_az_game
from chessckers_engine.variant_py import PyVariantClient

SEED = "8/8/3kkk2/8/8/8/3PPP2/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1"


def _model() -> ChesskersScorer:
    torch.manual_seed(0)
    return ChesskersScorer(d_hidden=64, c_filters=32, n_blocks=2).eval()


class _Counting:
    """Wraps a model, counting NN leaf evaluations (policy_and_value / value)."""

    def __init__(self, m: ChesskersScorer):
        self._m = m
        self.n = 0

    def policy_and_value(self, *a, **k):
        self.n += 1
        return self._m.policy_and_value(*a, **k)

    def value(self, *a, **k):
        self.n += 1
        return self._m.value(*a, **k)

    def parameters(self):
        return self._m.parameters()

    def __getattr__(self, k):
        return getattr(self._m, k)


def test_reuse_continues_correct_position_and_picks_legal_move():
    m, c = _model(), PyVariantClient()
    st0 = c.new_game(SEED)
    r0 = run_mcts(st0, c, m, n_sims=64, c_puct=1.5, dirichlet_alpha=None)
    child = r0.root.children[r0.chosen["uci"]]
    assert child.visits >= 1  # the played child carried real search

    st1 = c.make_move(st0["fen"], r0.chosen["uci"])
    r1 = run_mcts(st1, c, m, n_sims=64, c_puct=1.5, dirichlet_alpha=None, reuse_root=child)

    assert r1.root is child                          # reused the SAME node, re-rooted
    assert _node_fen(r1.root, c) == st1["fen"]        # onto the CORRECT position
    assert r1.root.visits >= 64                       # searched to the fixed budget
    legal = {mv["uci"] for mv in st1["legalMoves"]}
    assert r1.chosen["uci"] in legal                  # legal for the ACTUAL position


def test_mismatched_reuse_root_is_discarded():
    m, c = _model(), PyVariantClient()
    st0 = c.new_game(SEED)
    r0 = run_mcts(st0, c, m, n_sims=32, c_puct=1.5, dirichlet_alpha=None)
    # a child for a move OTHER than the chosen one -> a different position
    other = next(ch for uci, ch in r0.root.children.items() if uci != r0.chosen["uci"])
    st1 = c.make_move(st0["fen"], r0.chosen["uci"])
    assert _node_fen(other, c) != st1["fen"]          # genuinely the wrong position

    r1 = run_mcts(st1, c, m, n_sims=32, c_puct=1.5, dirichlet_alpha=None, reuse_root=other)
    assert r1.root is not other                       # discarded -> fresh root
    assert r1.chosen["uci"] in {mv["uci"] for mv in st1["legalMoves"]}


def test_full_game_with_reuse_completes_with_legal_moves():
    """play_az_game threads reuse every ply. A clean completion (no make_move
    abort) means every reused subtree matched its position."""
    os.environ["CHESSCKERS_START_FEN"] = SEED
    os.environ["CHESSCKERS_MAX_PLIES"] = "40"
    m, c = _model(), PyVariantClient()
    game = play_az_game(m, c, n_sims=32, temperature=1.0, temp_cutoff_plies=3,
                        max_plies=40, rng=torch.Generator().manual_seed(1))
    assert len(game.records) >= 1
    assert game.outcome in ("white", "black", "draw")


def test_reuse_reduces_nn_evaluations():
    """On the SAME position + same budget, a reused root does fewer NN evals than
    a fresh search (carried visits cover the shortfall)."""
    c = PyVariantClient()
    st0 = c.new_game(SEED)
    # Move 1 fresh, harvest the chosen child's subtree.
    m_warm = _Counting(_model())
    r0 = run_mcts(st0, c, m_warm, n_sims=64, c_puct=1.5, dirichlet_alpha=None)
    child = r0.root.children[r0.chosen["uci"]]
    st1 = c.make_move(st0["fen"], r0.chosen["uci"])

    # Move 2 WITH reuse (count only this move's new evals).
    m_warm.n = 0
    run_mcts(st1, c, m_warm, n_sims=64, c_puct=1.5, dirichlet_alpha=None, reuse_root=child)
    evals_reuse = m_warm.n

    # Move 2 FRESH on the same position (identical weights via the fixed seed).
    m_fresh = _Counting(_model())
    run_mcts(st1, c, m_fresh, n_sims=64, c_puct=1.5, dirichlet_alpha=None, reuse_root=None)
    evals_fresh = m_fresh.n

    assert evals_reuse < evals_fresh
