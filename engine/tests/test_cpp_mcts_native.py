"""Slice 6d test: the FULLY-NATIVE search (run_mcts_native: native move-gen +
apply + encode + NN forward, zero Python per leaf) must produce the EXACT same
visit distribution as the eval-callback search (run_mcts) when both use the same
native net — confirming the native eval path is wired correctly.

(Speed is a separate concern: the Slice-6 forward is plain-loop C++ for parity;
making it actually faster than PyTorch needs BLAS/Accelerate + batched leaves.)
"""
from __future__ import annotations

import pytest
import torch

from chessckers_engine.model import ChesskersScorer
from chessckers_engine.native_net import export_state_dict

cpp = pytest.importorskip("chessckers_cpp")

SEEDS = [
    "8/8/3kkk2/2PPPP2/2PPPP2/8/8/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
    "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
    "7p/8/8/8/8/8/8/4K3[h8:ssss] b - - 0 1",
    "8/8/3kkk2/2PPPP2/2PPPP2/8/8/4K3[d6:kk,e6:kk,f6:kk] w - - 0 1",
]


@pytest.fixture(scope="module")
def net(tmp_path_factory):
    torch.manual_seed(0)
    m = ChesskersScorer()
    m.eval()
    wpath = str(tmp_path_factory.mktemp("net") / "net.bin")
    export_state_dict(m.state_dict(), wpath)
    return cpp.ChesskersNet(wpath)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_sims", [16, 48])
def test_native_search_matches_eval_callback(net, seed: str, n_sims: int):
    def evf(fen, moves):
        b = cpp.parse_fen(fen)
        return net.eval(cpp.encode_position(b), [cpp.encode_move(mv) for mv in moves])

    _, vd_native, _ = cpp.run_mcts_native(cpp.parse_fen(seed), net, n_sims, 1.5)
    _, vd_callback = cpp.run_mcts(cpp.parse_fen(seed), evf, n_sims, 1.5)
    assert dict(vd_native) == dict(vd_callback), f"seed={seed} n_sims={n_sims}"
