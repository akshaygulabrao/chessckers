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

from chessckers_engine.model import ChesskersScorer, ChesskersScorerV2
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

    _, vd_native, *_ = cpp.run_mcts_native(cpp.parse_fen(seed), net, n_sims, 1.5)
    _, vd_callback = cpp.run_mcts(cpp.parse_fen(seed), evf, n_sims, 1.5)
    assert dict(vd_native) == dict(vd_callback), f"seed={seed} n_sims={n_sims}"


@pytest.fixture(scope="module")
def net_v2(tmp_path_factory):
    torch.manual_seed(0)
    m = ChesskersScorerV2(n_blocks=9, n_tf_blocks=7, n_heads=4, tf_ff_mult=4)
    m.eval()
    wpath = str(tmp_path_factory.mktemp("net_v2") / "net.bin")
    export_state_dict(m.state_dict(), wpath)
    return cpp.ChesskersNet(wpath)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_sims", [16, 48])
def test_native_v2_search_matches_eval_callback(net_v2, seed: str, n_sims: int):
    """The fully-native V2 search (run_mcts_native -> encode_pos/move_for -> V2
    encoders + gather-head eval) must equal the eval-callback search that encodes
    V2 explicitly — confirming run_mcts_native dispatches to the V2 encoders."""
    def evf(fen, moves):
        b = cpp.parse_fen(fen)
        return net_v2.eval(cpp.encode_position_v2(b), [cpp.encode_move_v2(mv) for mv in moves])

    _, vd_native, *_ = cpp.run_mcts_native(cpp.parse_fen(seed), net_v2, n_sims, 1.5)
    _, vd_callback = cpp.run_mcts(cpp.parse_fen(seed), evf, n_sims, 1.5)
    assert dict(vd_native) == dict(vd_callback), f"seed={seed} n_sims={n_sims}"


def test_native_tree_reuse_child_matches_pyvariant(net):
    """Tree reuse hinges on the detached child subtree's position matching the
    next search position. The native child's serialize_fen must equal the
    PyVariant fen after the same move — that equality is exactly the condition
    run_mcts_native checks before re-rooting — and the child must carry visits."""
    from chessckers_engine.variant_py import PyVariantClient

    seed = "7p/8/8/8/8/8/8/4K3[h8:ssss] b - - 0 1"
    chosen, _vd, _val, tree = cpp.run_mcts_native(cpp.parse_fen(seed), net, 64, 1.5)
    assert chosen, "search returned a move"
    child = tree.child(chosen)
    assert child is not None, "chosen child detaches into its own handle"
    assert child.visits() > 0, "reused subtree carried real search visits"
    nxt = PyVariantClient().make_move(seed, chosen)
    assert child.fen() == nxt["fen"], "native child fen == PyVariant fen -> reuse will hit"


def test_native_tree_reuse_tops_up_budget(net):
    """Reusing the child as the next root tops the SAME node up to n_sims — its
    carried visits count toward the budget (only the shortfall is searched)."""
    from chessckers_engine.variant_py import PyVariantClient

    seed = "7p/8/8/8/8/8/8/4K3[h8:ssss] b - - 0 1"
    chosen, _vd, _val, tree = cpp.run_mcts_native(cpp.parse_fen(seed), net, 64, 1.5)
    child = tree.child(chosen)
    assert child.visits() > 0
    nxt = PyVariantClient().make_move(seed, chosen)
    _c2, _vd2, _v2, tree2 = cpp.run_mcts_native(
        cpp.parse_fen(nxt["fen"]), net, 64, 1.5, 0.0, 0.25, 0, child)
    assert tree2.visits() == 64, "reused root searches up to the n_sims budget"


def test_native_selfplay_with_reuse_completes(net, monkeypatch):
    """A full native self-play game threading reuse_root every ply completes with a
    valid outcome — end-to-end exercise of detach + re-root + cross-boundary fen."""
    import torch

    from chessckers_engine.selfplay_az import play_az_game
    from chessckers_engine.native_search import make_native_search_fn
    from chessckers_engine.variant_py import PyVariantClient

    monkeypatch.setenv("CHESSCKERS_START_FEN", "7p/8/8/8/8/8/8/4K3[h8:ssss] b - - 0 1")
    search_fn = make_native_search_fn([net])
    game = play_az_game(None, PyVariantClient(), n_sims=32, search_fn=search_fn,
                        rng=torch.Generator().manual_seed(0),
                        dirichlet_alpha=0.5, dirichlet_eps=0.25, max_plies=60)
    assert game.outcome in ("white", "black", "draw")
    assert len(game.records) >= 1
