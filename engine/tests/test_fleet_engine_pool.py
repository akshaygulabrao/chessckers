"""Phase 3B-3b-ii (lc0-split cutover): fleet_client --spawn-engines drives N cc_selfplay
--jobs-local ENGINE procs instead of the Python selfplay_workers_only.

End-to-end (the real deployment shape, all subprocesses): a live in-process fleet_server
publishes a net; a fleet_client subprocess in --spawn-engines mode syncs weights.bin,
spawns a cc_selfplay engine, mints /next_game train jobs into its run-dir queue, the engine
claims+plays them off weights.bin and writes chunks, the client uploads them, and a server
STOP winds the whole thing down cleanly. Proves the kept orchestrator (HTTP/STOP/heartbeat)
owns the C++ engine binary — the lc0 client-owns-engine model with the engine swapped Py->C++.
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from chessckers_engine import fleet_server, train_continuous
from chessckers_engine.model import ChesskersScorer

pytest.importorskip("chessckers_cpp")

SEED_FEN = "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1"
CC_SELFPLAY = Path(__file__).resolve().parent.parent / "cpp" / "build" / "cc_selfplay"


@pytest.fixture
def server(tmp_path):
    rd = tmp_path / "srv"
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
        yield f"http://127.0.0.1:{httpd.server_address[1]}", rd, httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.slow
def test_spawn_engines_end_to_end(server, tmp_path):
    if not CC_SELFPLAY.exists():
        pytest.skip("cc_selfplay not built (run cpp/build.sh)")
    base, srv_rd, httpd = server
    train_continuous._publish(ChesskersScorer(), srv_rd / "weights.pt")  # .pt + .bin twins
    (srv_rd / "selfplay.json").write_text(json.dumps({"sims": 12, "max_plies": 20}))

    crd = tmp_path / "client"
    crd.mkdir()
    env = {**os.environ, "CHESSCKERS_START_FEN": SEED_FEN, "MACHINE": "enginebox"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "chessckers_engine.fleet_client",
         "--server", base, "--run-dir", str(crd), "--spawn-engines",
         "--engine-workers", "1", "--engine-worker-id-base", "500", "--engine-seed", "0",
         "--poll-seconds", "1", "--queue-depth", "2"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 90
        while time.time() < deadline and httpd.games_ingested < 1:
            if proc.poll() is not None:
                break
            time.sleep(0.5)
        assert httpd.games_ingested >= 1, "engine produced no uploaded games"
        # Server STOP -> /control returns STOP -> client stops the engine pool + exits.
        (srv_rd / "STOP").touch()
        out, _ = proc.communicate(timeout=45)
    finally:
        if proc.poll() is None:
            proc.kill()
            out, _ = proc.communicate()
    assert proc.returncode == 0, out
    # Uploaded chunks landed in the server buffer with the canonical (3+digit) name.
    pkls = sorted((srv_rd / "buffer").glob("*.pkl"))
    assert pkls, out
    assert fleet_server._NAME_RE.match(pkls[0].name), pkls[0].name
    # The engine synced + ran off the .bin the orchestrator distributed.
    assert (crd / "weights.bin").exists()
