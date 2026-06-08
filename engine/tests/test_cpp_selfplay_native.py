"""Phase 1 (lc0-split migration): the fully-native self-play game loop
(`cpp.play_game_native` — search+sample+apply+record per ply, zero Python in the
hot loop) must produce the EXACT same game as the Python `play_az_game` driving
the same native search, when both are deterministic (temperature=0, no Dirichlet,
no resign). This isolates the LOOP logic (terminal detection, visit alignment,
argmax move choice, apply, record) from RNG — the search itself is already
parity-locked by test_cpp_mcts_native.
"""
from __future__ import annotations

import pytest
import torch

from chessckers_engine.model import ChesskersScorer, ChesskersScorerV2
from chessckers_engine.native_net import export_state_dict

cpp = pytest.importorskip("chessckers_cpp")

SEEDS = [
    "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
    "7p/8/8/8/8/8/8/4K3[h8:ssss] b - - 0 1",
]


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


def _assert_games_identical(native_game, py_game):
    assert native_game.outcome == py_game.outcome
    assert len(native_game.records) == len(py_game.records)
    for a, b in zip(native_game.records, py_game.records):
        assert a.fen == b.fen
        assert a.side_to_move == b.side_to_move
        da = {m["uci"]: v for m, v in zip(a.legal_moves, a.visit_counts)}
        db = {m["uci"]: v for m, v in zip(b.legal_moves, b.visit_counts)}
        assert da == db, f"visit dists differ at fen {a.fen}"


def _run_parity(net_obj, start, monkeypatch, n_sims=32, max_plies=40):
    from chessckers_engine.native_search import make_native_search_fn, play_game_native
    from chessckers_engine.selfplay_az import play_az_game
    from chessckers_engine.variant_py import PyVariantClient

    # Python reference: play_az_game driving the native search, fully deterministic.
    monkeypatch.setenv("CHESSCKERS_START_FEN", start)
    py_game = play_az_game(
        None, PyVariantClient(), n_sims=n_sims, c_puct=1.5, temperature=0.0,
        temp_cutoff_plies=0, dirichlet_alpha=None, max_plies=max_plies,
        search_fn=make_native_search_fn([net_obj]),
    )
    # Native loop, same settings, deterministic (temp=0, no noise, no resign).
    native_game = play_game_native(
        net_obj, start_fen=start, n_sims=n_sims, c_puct=1.5, temperature=0.0,
        temp_cutoff_plies=0, max_plies=max_plies, dirichlet_alpha=0.0, resign_threshold=0.0,
    )
    return native_game, py_game


@pytest.mark.parametrize("start", SEEDS)
def test_native_game_loop_matches_python_v1(net, start, monkeypatch):
    _assert_games_identical(*_run_parity(net, start, monkeypatch))


@pytest.mark.parametrize("start", SEEDS)
def test_native_game_loop_matches_python_v2(net_v2, start, monkeypatch):
    _assert_games_identical(*_run_parity(net_v2, start, monkeypatch))


def test_native_game_loop_records_feed_examples(net, monkeypatch):
    """The native records adapt cleanly into the existing training pipeline:
    az_game_to_examples produces one AZExample per recorded ply."""
    from chessckers_engine.native_search import play_game_native
    from chessckers_engine.selfplay_az import az_game_to_examples

    start = SEEDS[0]
    game = play_game_native(net, start_fen=start, n_sims=24, temperature=1.0,
                            dirichlet_alpha=0.3, seed=7, max_plies=40)
    assert game.outcome in ("white", "black", "draw")
    assert len(game.records) >= 1
    examples = az_game_to_examples(game)
    assert len(examples) == len(game.records)
    ex = examples[0]
    assert abs(sum(ex.visit_distribution) - 1.0) < 1e-5
    assert len(ex.wdl_target) == 3
