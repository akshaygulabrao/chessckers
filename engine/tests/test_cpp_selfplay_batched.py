"""Phase 6d (lc0-split migration): GPU leaf batching. `cpp.play_games_batched_native`
runs `num_games` games across `batch_size` concurrent threads that SHARE one batched
forward per round (the lc0 design: many games, one batched backend). The CPU forward
(net.eval_batch) is byte-identical to K serial eval() calls, so each game stays
byte-identical to the single-threaded reference:

  play_games_batched_native(..., num_games=G, batch_size=B, base_seed=S, use_gpu=False)[i]
      ==  play_game_native(..., seed=S+i)

for every i, INDEPENDENT of B. This proves the batching is a pure inference-transport
optimization — it does NOT change search semantics, so the byte-parity gate that backs
the whole C++ port is preserved (not loosened). Determinism is forced with temperature>0
+ Dirichlet ON (the per-game rng stream must reproduce exactly under batched threading).

The `use_gpu=True` path (Apple Metal) is float-close, not byte-identical (GPU float32 vs
the CPU BLAS trunk), so it gets a separate smoke check rather than the parity assertion.
"""
from __future__ import annotations

import sys

import pytest
import torch

from chessckers_engine.model import ChesskersScorer, ChesskersScorerV2
from chessckers_engine.native_net import export_state_dict

cpp = pytest.importorskip("chessckers_cpp")

SEED_FEN = "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1"


def _bin(m, tmp_path_factory, name):
    m.eval()
    wpath = str(tmp_path_factory.mktemp(name) / "net.bin")
    export_state_dict(m.state_dict(), wpath)
    return cpp.ChesskersNet(wpath)


@pytest.fixture(scope="module")
def net(tmp_path_factory):
    torch.manual_seed(0)
    return _bin(ChesskersScorer(), tmp_path_factory, "net")


@pytest.fixture(scope="module")
def net_v2(tmp_path_factory):
    torch.manual_seed(0)
    return _bin(ChesskersScorerV2(n_blocks=2, n_tf_blocks=1, n_heads=4, tf_ff_mult=2),
                tmp_path_factory, "net_v2")


def _assert_game_tuple_eq(a, b):
    a_records, a_outcome, a_final = a
    b_records, b_outcome, b_final = b
    assert a_outcome == b_outcome
    assert a_final == b_final
    assert len(a_records) == len(b_records)
    for (afen, alegal, avc, aside), (bfen, blegal, bvc, bside) in zip(a_records, b_records):
        assert afen == bfen
        assert aside == bside
        assert list(avc) == list(bvc)
        assert [m["uci"] for m in alegal] == [m["uci"] for m in blegal]
        assert list(alegal) == list(blegal)


def _params(**over):
    p = dict(n_sims=24, c_puct=1.5, temperature=1.0, temp_cutoff_plies=8, max_plies=40,
             dirichlet_alpha=0.3, dirichlet_eps=0.25)
    p.update(over)
    return p


@pytest.mark.parametrize("batch_size", [1, 4])
def test_batched_matches_single_v1(net, batch_size):
    G, S = 6, 100
    p = _params()
    batch = cpp.play_games_batched_native(cpp.parse_fen(SEED_FEN), net, num_games=G,
                                          batch_size=batch_size, base_seed=S, use_gpu=False, **p)
    assert len(batch) == G
    for i in range(G):
        single = cpp.play_game_native(cpp.parse_fen(SEED_FEN), net, seed=S + i, **p)
        _assert_game_tuple_eq(single, batch[i])


@pytest.mark.parametrize("batch_size", [1, 4])
def test_batched_matches_single_v2(net_v2, batch_size):
    G, S = 6, 100
    p = _params()
    batch = cpp.play_games_batched_native(cpp.parse_fen(SEED_FEN), net_v2, num_games=G,
                                          batch_size=batch_size, base_seed=S, use_gpu=False, **p)
    assert len(batch) == G
    for i in range(G):
        single = cpp.play_game_native(cpp.parse_fen(SEED_FEN), net_v2, seed=S + i, **p)
        _assert_game_tuple_eq(single, batch[i])


def test_batch_size_invariance(net_v2):
    """The batch output is identical regardless of the batch width (1 vs 8 concurrent)."""
    G, S = 8, 7
    p = _params()
    b1 = cpp.play_games_batched_native(cpp.parse_fen(SEED_FEN), net_v2, num_games=G, batch_size=1,
                                       base_seed=S, use_gpu=False, **p)
    b8 = cpp.play_games_batched_native(cpp.parse_fen(SEED_FEN), net_v2, num_games=G, batch_size=8,
                                       base_seed=S, use_gpu=False, **p)
    for i in range(G):
        _assert_game_tuple_eq(b1[i], b8[i])


def test_more_games_than_batch_slots(net_v2):
    """num_games > batch_size: workers refill from the queue; every game still matches."""
    G, B, S = 10, 3, 50
    p = _params()
    batch = cpp.play_games_batched_native(cpp.parse_fen(SEED_FEN), net_v2, num_games=G,
                                          batch_size=B, base_seed=S, use_gpu=False, **p)
    assert len(batch) == G
    for i in range(G):
        single = cpp.play_game_native(cpp.parse_fen(SEED_FEN), net_v2, seed=S + i, **p)
        _assert_game_tuple_eq(single, batch[i])


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal GPU backend is Apple-only")
def test_gpu_batched_runs_and_is_close(net_v2):
    """use_gpu=True (Metal): float-close, not byte-identical. Smoke + soft visit check."""
    if not hasattr(cpp, "MetalTrunkV2") or not cpp.MetalTrunkV2(net_v2).ok():
        pytest.skip("no Metal GPU on this box")
    G, B, S = 6, 6, 100
    p = _params(dirichlet_alpha=0.0)  # drop root noise so divergence is purely GPU-float
    gpu = cpp.play_games_batched_native(cpp.parse_fen(SEED_FEN), net_v2, num_games=G,
                                        batch_size=B, base_seed=S, use_gpu=True, **p)
    assert len(gpu) == G
    for records, outcome, _final in gpu:
        assert outcome in ("white", "black", "draw")
        assert len(records) > 0
    # First move of game 0 should pick the same top-visited move as the CPU forward
    # (per-eval parity is ~1e-4; the argmax visit is robust to that).
    cpu = cpp.play_game_native(cpp.parse_fen(SEED_FEN), net_v2, seed=S, **p)
    g_rec0, c_rec0 = gpu[0][0], cpu[0]
    g_top = max(zip(g_rec0[0][2], (m["uci"] for m in g_rec0[0][1])))[1]
    c_top = max(zip(c_rec0[0][2], (m["uci"] for m in c_rec0[0][1])))[1]
    assert g_top == c_top
