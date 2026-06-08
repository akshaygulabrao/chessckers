"""Phase 3B-2b (lc0-split): the standalone C++ self-play client loop.

  - cpp.parse_job        the C++ next_game job-JSON parse (cross-checked vs json.loads)
  - cpp.run_selfplay_client  the full loop (next_game -> get_network[cache by sha] ->
    play_game_chunk -> upload_game) against a LIVE in-process Python fleet_server
  - cc_selfplay          the standalone executable (NO Python) doing the same, run as
    a subprocess against the live server (marked slow)

The client uploads ccz chunks the Python server buffers and decode_chunk recovers;
the .meta carries the same {worker_id, machine, outcome, plies, seed_fen} shape the
Python worker wrote. Train jobs only — gate/match play in C++ is Phase 4.
"""
from __future__ import annotations

import itertools
import json
import subprocess
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from chessckers_engine import fleet_server, train_continuous
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.training_chunk import decode_chunk

cpp = pytest.importorskip("chessckers_cpp")

SEED_FEN = "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1"
CC_SELFPLAY = Path(__file__).resolve().parent.parent / "cpp" / "build" / "cc_selfplay"


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


# --- the job-JSON parser -------------------------------------------------------

def test_parse_job_train():
    body = json.dumps({
        "type": "train", "sha": "aa", "bin_sha": "bb",
        "params": {"sims": 64, "c_puct": 1.25, "temperature": 0.8, "max_plies": 200,
                   "dirichlet_alpha": 0.3, "dirichlet_eps": 0.25, "unknown_future_key": 9},
    })
    j = cpp.parse_job(body.encode())
    assert j["type"] == "train" and j["sha"] == "aa" and j["bin_sha"] == "bb"
    assert j["params"]["n_sims"] == 64           # "sims" -> n_sims
    assert j["params"]["c_puct"] == 1.25
    assert j["params"]["temperature"] == 0.8
    assert j["params"]["max_plies"] == 200
    assert j["params"]["temp_cutoff_plies"] == 30  # absent -> default


def test_parse_job_defaults_when_params_absent():
    j = cpp.parse_job(json.dumps({"type": "train", "bin_sha": ""}).encode())
    assert j["params"]["n_sims"] == 100 and j["params"]["resign_threshold"] == 0.0


# --- the full client loop (in-process) -----------------------------------------

def test_client_loop_plays_and_uploads(server, tmp_path):
    base, rd = server
    train_continuous._publish(ChesskersScorer(), rd / "weights.pt")
    (rd / "selfplay.json").write_text(json.dumps({"sims": 12, "max_plies": 20}))
    cache = tmp_path / "netcache"
    cache.mkdir()

    n = cpp.run_selfplay_client(base, SEED_FEN, num_games=3, worker_id=300,
                                machine="testbox", base_seed=1, net_cache_dir=str(cache))
    assert n == 3
    pkls = sorted((rd / "buffer").glob("*.pkl"))
    assert len(pkls) == 3
    for pkl in pkls:
        assert decode_chunk(pkl.read_bytes())                     # valid ccz chunk
        meta = json.loads((Path(str(pkl) + ".meta")).read_text())
        assert meta["worker_id"] == 300 and meta["machine"] == "testbox"
        assert meta["outcome"] in ("white", "black", "draw") and meta["plies"] >= 1
    # the net was cached once (one bin_sha), not re-downloaded per game
    assert len(list(cache.glob("net-*.bin"))) == 1


def test_client_loop_noops_without_bin(server, tmp_path):
    """A run with a .pt but no .bin (bin_sha='') uploads nothing — additive safety."""
    base, rd = server
    (rd / "weights.pt").write_bytes(b"W")  # no .bin twin
    cache = tmp_path / "nc"
    cache.mkdir()
    n = cpp.run_selfplay_client(base, SEED_FEN, num_games=2, net_cache_dir=str(cache))
    assert n == 0 and not list((rd / "buffer").glob("*.pkl"))


# --- the standalone executable (no Python) -------------------------------------

@pytest.mark.slow
def test_cc_selfplay_executable(server, tmp_path):
    if not CC_SELFPLAY.exists():
        pytest.skip("cc_selfplay not built (run cpp/build.sh)")
    base, rd = server
    train_continuous._publish(ChesskersScorer(), rd / "weights.pt")
    cache = tmp_path / "exe_cache"
    cache.mkdir()
    proc = subprocess.run(
        [str(CC_SELFPLAY), "--server", base, "--games", "2", "--worker-id", "401",
         "--machine", "exebox", "--seed", "3", "--cache-dir", str(cache),
         "--start-fen", SEED_FEN],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "uploaded 2 games" in proc.stdout
    pkls = sorted((rd / "buffer").glob("*.pkl"))
    assert len(pkls) == 2
    for pkl in pkls:
        assert decode_chunk(pkl.read_bytes())
        assert json.loads(Path(str(pkl) + ".meta").read_text())["machine"] == "exebox"
