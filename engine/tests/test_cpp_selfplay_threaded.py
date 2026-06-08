"""Phase 2 (lc0-split migration): the pure-C++, multi-threaded self-play driver
(`cpp.play_games_native`) must produce games byte-identical to the Phase-1
single-threaded `cpp.play_game_native`. Game i is seeded base_seed+i, so:

  play_games_native(..., num_games=G, num_threads=T, base_seed=S)[i]
      ==  play_game_native(..., seed=S+i)

for every i, INDEPENDENT of T. This proves (a) the pure-C++ move plumbing
(NativeMove apply/encode) matches the dict path exactly, and (b) threading is
deterministic — no shared-state races across the worker threads (one net shared
read-only). Determinism is forced with temperature>0 + Dirichlet ON: the per-game
rng stream must be reproduced exactly under threading, a strictly stronger check
than the temp=0 path.
"""
from __future__ import annotations

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
    """a, b are (records, outcome, final_status) as returned by play_game_native /
    one entry of play_games_native."""
    a_records, a_outcome, a_final = a
    b_records, b_outcome, b_final = b
    assert a_outcome == b_outcome
    assert a_final == b_final
    assert len(a_records) == len(b_records)
    for (afen, alegal, avc, aside), (bfen, blegal, bvc, bside) in zip(a_records, b_records):
        assert afen == bfen
        assert aside == bside
        assert list(avc) == list(bvc)
        # legal-move lists: same order, same uci, same full dicts
        assert [m["uci"] for m in alegal] == [m["uci"] for m in blegal]
        assert list(alegal) == list(blegal)


def _params(**over):
    p = dict(n_sims=24, c_puct=1.5, temperature=1.0, temp_cutoff_plies=8, max_plies=40,
             dirichlet_alpha=0.3, dirichlet_eps=0.25)
    p.update(over)
    return p


@pytest.mark.parametrize("num_threads", [1, 4])
def test_play_games_native_matches_single_v1(net, num_threads):
    G, S = 6, 100
    p = _params()
    batch = cpp.play_games_native(cpp.parse_fen(SEED_FEN), net, num_games=G,
                                  num_threads=num_threads, base_seed=S, **p)
    assert len(batch) == G
    for i in range(G):
        single = cpp.play_game_native(cpp.parse_fen(SEED_FEN), net, seed=S + i, **p)
        _assert_game_tuple_eq(single, batch[i])


@pytest.mark.parametrize("num_threads", [1, 4])
def test_play_games_native_matches_single_v2(net_v2, num_threads):
    G, S = 6, 100
    p = _params()
    batch = cpp.play_games_native(cpp.parse_fen(SEED_FEN), net_v2, num_games=G,
                                  num_threads=num_threads, base_seed=S, **p)
    assert len(batch) == G
    for i in range(G):
        single = cpp.play_game_native(cpp.parse_fen(SEED_FEN), net_v2, seed=S + i, **p)
        _assert_game_tuple_eq(single, batch[i])


def test_thread_count_invariance(net_v2):
    """The batch output is identical regardless of how many threads ran it."""
    G, S = 8, 7
    p = _params()
    b1 = cpp.play_games_native(cpp.parse_fen(SEED_FEN), net_v2, num_games=G, num_threads=1,
                               base_seed=S, **p)
    b8 = cpp.play_games_native(cpp.parse_fen(SEED_FEN), net_v2, num_games=G, num_threads=8,
                               base_seed=S, **p)
    for i in range(G):
        _assert_game_tuple_eq(b1[i], b8[i])


def test_batch_records_feed_examples(net_v2):
    """A batched game adapts into the existing training pipeline unchanged."""
    from chessckers_engine.selfplay_az import AZGame, AZRecord, az_game_to_examples

    batch = cpp.play_games_native(cpp.parse_fen(SEED_FEN), net_v2, num_games=2, num_threads=2,
                                  base_seed=1, **_params())
    records_raw, outcome, final_status = batch[0]
    records = [AZRecord(fen=fen, legal_moves=list(legal), visit_counts=list(vc), side_to_move=side)
               for (fen, legal, vc, side) in records_raw]
    game = AZGame(records=records, final_status=final_status, outcome=outcome)
    examples = az_game_to_examples(game)
    assert len(examples) == len(records)
    assert abs(sum(examples[0].visit_distribution) - 1.0) < 1e-5
