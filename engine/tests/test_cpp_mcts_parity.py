"""Slice 5b/5c oracle test: the native C++ PUCT search must produce the EXACT
same root visit distribution as the Python mcts_puct.run_mcts, given the same
network and no Dirichlet noise (fully deterministic).

The C++ tree does selection / expansion (native apply) / backup; only the leaf
NN forward crosses into Python via eval_fn = _eval_and_priors (the same call
Python's run_mcts uses). Identical priors + values + PUCT math ⇒ identical tree
⇒ identical visit counts. Seeds are endgame positions (no castling rights, no
en passant) so the C++ serialize→encode path and Python's state→encode path are
bit-identical.
"""
from __future__ import annotations

import pytest
import torch

from chessckers_engine.mcts_puct import _eval_and_priors, run_mcts
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.variant_py.client import PyVariantClient

cpp = pytest.importorskip("chessckers_cpp")

SEEDS = [
    "8/8/3kkk2/2PPPP2/2PPPP2/8/8/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
    "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
    "7p/8/8/8/8/8/8/4K3[h8:ssss] b - - 0 1",
    "8/8/3kkk2/2PPPP2/2PPPP2/8/8/4K3[d6:kk,e6:kk,f6:kk] w - - 0 1",
]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = ChesskersScorer()
    m.eval()
    return m


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_sims", [16, 48])
def test_run_mcts_parity_vs_python(model, seed: str, n_sims: int):
    client = PyVariantClient()

    def eval_fn(fen, legal_moves):
        return _eval_and_priors(model, fen, list(legal_moves))

    chosen_cpp, vd_cpp = cpp.run_mcts(cpp.parse_fen(seed), eval_fn, n_sims, 1.5)

    st = client.new_game(seed)
    res = run_mcts(st, client, model, n_sims=n_sims, c_puct=1.5, dirichlet_alpha=None)

    assert dict(vd_cpp) == res.visit_distribution, (
        f"\nseed={seed} n_sims={n_sims}\n cpp={dict(vd_cpp)}\n  py={res.visit_distribution}"
    )
    assert chosen_cpp == res.chosen["uci"], f"chosen mismatch seed={seed}"
    # sanity: visits sum to n_sims-1 (the first sim expands the root, visiting no child)
    assert sum(dict(vd_cpp).values()) == n_sims - 1
