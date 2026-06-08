"""Phase 3B-1 (lc0-split): the fleet server publishes a C++-loadable native `.bin`
net so a NO-PYTHON self-play client can fetch a usable net by content address and
play — the keystone that unblocks a standalone C++ client (cpp-httplib loop = 3B-2).

lc0 decision copied: the training server serves CLIENT-READY weights. Here that's
an additive `.bin` twin of the served `.pt`:
  - train_continuous._publish writes weights.bin beside weights.pt (atomic);
  - fleet_server serves .bin by sha (GET /get_network) and carries `bin_sha` in
    the POST /next_game train job;
  - the .pt path (Python clients) is untouched.

End-to-end gate: POST /next_game -> bin_sha -> GET /get_network?sha=bin_sha ->
bytes load into cpp.ChesskersNet and play a game -> ccz chunk decodes. No torch
on the "client" side of the fetch; the bytes ARE the network.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from chessckers_engine import fleet_server, train_continuous
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.training_chunk import decode_chunk

cpp = pytest.importorskip("chessckers_cpp")

SEED_FEN = "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1"


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


def _get(url: str):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read()


def _post(url: str, data: bytes):
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, r.read()


def test_publish_writes_bin_twin(tmp_path):
    """The trainer publishes weights.bin alongside weights.pt, loadable by the C++ net."""
    rd = tmp_path / "run"
    rd.mkdir()
    train_continuous._publish(ChesskersScorer(), rd / "weights.pt")
    assert (rd / "weights.pt").exists()
    bin_path = rd / "weights.bin"
    assert bin_path.exists()
    # the bytes ARE a loadable native net (no torch needed to consume them)
    cpp.ChesskersNet(str(bin_path))


def test_next_game_carries_bin_sha(server):
    base, rd = server
    train_continuous._publish(ChesskersScorer(), rd / "weights.pt")
    pt_sha = hashlib.sha256((rd / "weights.pt").read_bytes()).hexdigest()
    bin_sha = hashlib.sha256((rd / "weights.bin").read_bytes()).hexdigest()

    _, body = _post(f"{base}/next_game", b"")
    job = json.loads(body)
    assert job["type"] == "train"
    assert job["sha"] == pt_sha          # .pt path unchanged (Python clients)
    assert job["bin_sha"] == bin_sha     # additive C++-client net address


def test_bin_sha_empty_before_publish(server):
    """No .bin yet -> bin_sha is '' (additive: never breaks the .pt train job)."""
    base, rd = server
    (rd / "weights.pt").write_bytes(b"W")  # a .pt with no .bin twin
    _, body = _post(f"{base}/next_game", b"")
    job = json.loads(body)
    assert job["bin_sha"] == ""
    assert job["sha"] == hashlib.sha256(b"W").hexdigest()


def test_no_python_client_fetches_and_plays(server, tmp_path):
    """Full keystone: a client with no torch gets bin_sha from /next_game, fetches the
    net by content address, loads it, plays a game, and the ccz chunk round-trips."""
    base, rd = server
    train_continuous._publish(ChesskersScorer(), rd / "weights.pt")

    _, body = _post(f"{base}/next_game", b"")
    job = json.loads(body)
    bin_sha = job["bin_sha"]
    assert bin_sha

    status, net_bytes = _get(f"{base}/get_network?sha={bin_sha}")
    assert status == 200
    assert hashlib.sha256(net_bytes).hexdigest() == bin_sha  # content-addressed integrity

    net_file = tmp_path / "fetched.bin"
    net_file.write_bytes(net_bytes)
    net = cpp.ChesskersNet(str(net_file))  # loads the served bytes directly — no torch

    chunk = cpp.play_game_chunk(cpp.parse_fen(SEED_FEN), net, n_sims=16, temperature=1.0,
                                temp_cutoff_plies=4, max_plies=24, dirichlet_alpha=0.3, seed=1)
    examples = decode_chunk(chunk)
    assert examples and all(abs(sum(e.visit_distribution) - 1.0) < 1e-6 for e in examples)
