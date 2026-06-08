"""Phase 3B-3b-i (lc0-split orchestrator+engine): the orchestrator distributes the
C++-loadable net so a cc_selfplay --jobs-local engine can run off the run-dir.

Additive to the running Python fleet (it still syncs weights.pt + reads cand_path):
  - fleet_server /control advertises X-Network-Bin-Sha (the .bin twin's content address);
  - fleet_client._pull_sha_to materializes run-dir/weights.bin off it;
  - fleet_client._fetch_net(..., ".bin") pulls gate .bin twins (cand_bin/opp_bin).

Gate (end-to-end): server publishes -> client syncs weights.bin -> cpp.run_jobs_local
plays a minted train job off that synced net -> the chunk decodes. Proves the orchestrator
hands the engine everything it needs with NO change to how Python workers spawn.
"""
from __future__ import annotations

import itertools
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from chessckers_engine import fleet_client, fleet_server, train_continuous
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.training_chunk import decode_chunk

cpp = pytest.importorskip("chessckers_cpp")

SEED_FEN = "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1"


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
        yield f"http://127.0.0.1:{httpd.server_address[1]}", rd
    finally:
        httpd.shutdown()
        httpd.server_close()


def _control_headers(base: str) -> dict:
    with urllib.request.urlopen(f"{base}/control", timeout=10) as r:
        return dict(r.headers)


def test_control_advertises_bin_sha_only_after_publish(server):
    base, rd = server
    # Before any net: additive header present but empty.
    assert _control_headers(base).get("X-Network-Bin-Sha", "") == ""
    train_continuous._publish(ChesskersScorer(), rd / "weights.pt")
    assert (rd / "weights.bin").exists()
    bin_sha = _control_headers(base).get("X-Network-Bin-Sha", "")
    assert bin_sha == fleet_server._file_sha(rd / "weights.bin") != ""


def test_client_syncs_weights_bin_and_engine_plays(server, tmp_path):
    base, rd = server
    train_continuous._publish(ChesskersScorer(), rd / "weights.pt")
    want_bin = _control_headers(base)["X-Network-Bin-Sha"]

    # The orchestrator sync (the new step-2 .bin branch), exercised directly.
    crd = tmp_path / "client"
    (crd / "jobs").mkdir(parents=True)
    have = fleet_client._pull_sha_to(base, crd / "weights.bin", want_bin, "", 10.0, "bin")
    assert have == want_bin
    assert (crd / "weights.bin").read_bytes() == (rd / "weights.bin").read_bytes()
    # No-op when unchanged (sha already on disk).
    assert fleet_client._pull_sha_to(base, crd / "weights.bin", want_bin, have, 10.0, "bin") == have

    # End-to-end: the engine plays a minted train job off the synced weights.bin.
    (crd / "jobs" / "0.json").write_text(json.dumps(
        {"type": "train", "params": {"sims": 16, "max_plies": 30}}))
    handled = cpp.run_jobs_local(str(crd), SEED_FEN, worker_id=400, machine="testbox",
                                 base_seed=1, max_jobs=1)
    assert handled == 1
    pkls = list((crd / "buffer").glob("*.pkl"))
    assert len(pkls) == 1 and decode_chunk(pkls[0].read_bytes())


def test_fetch_net_pulls_bin_twin(server, tmp_path):
    base, rd = server
    train_continuous._publish(ChesskersScorer(), rd / "weights.pt")
    bin_sha = _control_headers(base)["X-Network-Bin-Sha"]
    gate = tmp_path / "_gate"
    p = fleet_client._fetch_net(base, bin_sha, gate, 10.0, ".bin")
    assert p is not None and p.suffix == ".bin"
    cpp.ChesskersNet(str(p))  # loadable C++ net
