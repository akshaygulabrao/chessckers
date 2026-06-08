"""Phase 4b (lc0-split): C++ two-net GATE match play + the client's match handling.

  - cpp.parse_job        parses a "match" job's fields (cross-checked vs json.loads)
  - cpp.play_match_game  one gate game (white_net vs black_net) — BYTE-IDENTICAL
    (outcome + full move list) to the Python gate path (fleet_arena._play_from driven
    by _native_picker) for V1+V2, with Dirichlet noise ON (RNG-stream parity, strictly
    stronger than temp=0) and across several start positions
  - cpp.run_selfplay_client  now also plays match jobs end-to-end against a LIVE
    in-process Python fleet_server and POSTs /match_result; the outcomes land in
    match_results/ and fleet_arena._GateCollector tallies them (the gate would promote)

The arena stays Python (it plays no game — dispatch + tally only); only the gate game
PLAY moves to C++, exactly like the self-play path did in 3B.
"""
from __future__ import annotations

import itertools
import json
import threading
from http.server import ThreadingHTTPServer

import pytest
import torch

from chessckers_engine import fleet_arena, fleet_server
from chessckers_engine.fleet_arena import _GateCollector, _score_opp
from chessckers_engine.model import build_model
from chessckers_engine.native_net import export_state_dict
from chessckers_engine.variant_py import PyVariantClient

cpp = pytest.importorskip("chessckers_cpp")

# Black to move, a small reachable endgame — terminates well within max_plies.
SEED_FEN = "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1"
OPEN_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.fixture
def server(tmp_path):
    rd = tmp_path / "run"
    (rd / "match_nets").mkdir(parents=True)
    (rd / "buffer").mkdir(parents=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), fleet_server._Handler)
    httpd.run_dir = rd
    httpd.match_cursor = itertools.count()
    httpd.result_counter = itertools.count()
    httpd.clients = {}
    httpd.clients_lock = threading.Lock()
    httpd.games_ingested = 0
    httpd.stats_lock = threading.Lock()
    httpd.code_version = "test"
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", rd
    finally:
        httpd.shutdown()
        httpd.server_close()


def _make_cpp_net(tmp_path, name, arch, seed):
    """Two distinctly-initialised cpp.ChesskersNet — the candidate vs the opponent."""
    torch.manual_seed(seed)
    m = build_model(**arch)
    m.eval()
    binp = tmp_path / f"{name}.bin"
    export_state_dict(m.state_dict(), binp)
    return cpp.ChesskersNet(str(binp))


def _python_match(white_net, black_net, start_fen, sims, c_puct, dir_alpha, dir_eps,
                  max_plies, base):
    """The Python gate path: fleet_arena._play_from driven by _native_picker, with a
    shared per-move Dirichlet counter (as MatchRunner uses) and the chosen ucis
    recorded so we can compare the full move list, not just the outcome."""
    counter = itertools.count(base)
    moves: list[str] = []
    client = PyVariantClient()

    def wrap(net):
        pick = fleet_arena._native_picker(net, sims, c_puct, dir_alpha, dir_eps, counter)

        def p(state):
            m = pick(state)
            if m is not None:
                moves.append(m["uci"])
            return m
        return p

    outcome = fleet_arena._play_from(wrap(white_net), wrap(black_net), client, start_fen, max_plies)
    return outcome, moves


# --- the job parser ------------------------------------------------------------

def test_parse_job_match():
    body = json.dumps({
        "type": "match", "match_id": 7, "sha": "c", "candidate_sha": "c",
        "opponent": "net-100", "opponent_sha": "o",
        "candidate_bin_sha": "cb", "opponent_bin_sha": "ob",
        "seed": SEED_FEN, "cand_white": False, "arch": {"version": "v1"},
        "params": {"sims": 40, "c_puct": 1.1, "dir_alpha": 0.3, "dir_eps": 0.2, "max_plies": 50},
    })
    j = cpp.parse_job(body.encode())
    assert j["type"] == "match"
    m = j["match"]
    assert m["match_id"] == 7 and m["opponent"] == "net-100"
    assert m["candidate_bin_sha"] == "cb" and m["opponent_bin_sha"] == "ob"
    assert m["seed"] == SEED_FEN and m["cand_white"] is False
    assert m["sims"] == 40 and m["c_puct"] == 1.1
    assert m["dir_alpha"] == 0.3 and m["dir_eps"] == 0.2 and m["max_plies"] == 50


def test_parse_job_match_absent_when_train():
    j = cpp.parse_job(json.dumps({"type": "train", "bin_sha": ""}).encode())
    assert "match" not in j  # match block only populated for match jobs


# --- two-net match play parity (the gate) --------------------------------------

@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("start_fen", [SEED_FEN, OPEN_FEN])
@pytest.mark.parametrize("dir_alpha", [0.0, 0.3])
def test_play_match_game_parity(tmp_path, version, start_fen, dir_alpha):
    """C++ play_match_game == Python _play_from(_native_picker) — same (outcome, moves).
    With Dirichlet ON the per-move RNG streams must align move-for-move (stronger than
    a temp=0 deterministic check)."""
    arch = {"version": version}
    white = _make_cpp_net(tmp_path, "white", arch, seed=11)
    black = _make_cpp_net(tmp_path, "black", arch, seed=22)
    sims, c_puct, dir_eps, max_plies, base = 16, 1.5, 0.25, 24, 1

    py_outcome, py_moves = _python_match(white, black, start_fen, sims, c_puct, dir_alpha,
                                         dir_eps, max_plies, base)
    cpp_outcome, cpp_moves = cpp.play_match_game(white, black, start_fen, sims, c_puct,
                                                 dir_alpha, dir_eps, max_plies, base)
    assert cpp_outcome == py_outcome
    assert list(cpp_moves) == py_moves


# --- end-to-end: a C++ client plays the gate, the arena tallies it -------------

def _save_net(path, arch):
    m = build_model(**arch)
    m.eval()
    torch.save(m.state_dict(), path)


def test_client_plays_gate_and_arena_tallies(server, tmp_path):
    base, rd = server
    arch = {"version": "v1"}
    for name in ("cand.pt", "best.pt"):
        _save_net(rd / name, arch)
        fleet_arena._publish_gate_bin(rd / name, arch)
    (rd / "match.json").write_text(json.dumps({
        "match_id": 1, "seeds": [SEED_FEN], "opponents": ["best"], "arch": arch,
        "params": {"sims": 8, "c_puct": 1.5, "dir_alpha": 0.3, "dir_eps": 0.25, "max_plies": 12},
    }))
    cache = tmp_path / "netcache"
    cache.mkdir()

    # opp x seed x side = 1*1*2 = 2 units; 4 games -> 2 per side.
    n = cpp.run_selfplay_client(base, SEED_FEN, num_games=4, worker_id=500, machine="gatebox",
                                base_seed=7, net_cache_dir=str(cache))
    assert n == 4

    results = sorted((rd / "match_results").glob("1_*.json"))
    assert len(results) == 4
    for p in results:
        r = json.loads(p.read_text())
        assert r["outcome"] in ("white", "black", "draw")
        assert r["opp"] == "best" and r["match_id"] == 1
        assert r["seed"] == SEED_FEN

    # the arena's tally drains those exact files (it would score the gate from this).
    gc = _GateCollector(rd / "match_results", 1)
    assert gc.have(["best"], [SEED_FEN], pairs=2) == 4
    sc = _score_opp(gc.collected_for("best", [SEED_FEN]), [SEED_FEN], pairs=2)
    assert 0.0 <= sc["score"] <= 1.0
    assert sc["record"]["w"] + sc["record"]["l"] + sc["record"]["d"] == 4
    # both nets were fetched by content address and cached (cand + opp = 2 .bin files).
    assert len(list(cache.glob("net-*.bin"))) == 2
