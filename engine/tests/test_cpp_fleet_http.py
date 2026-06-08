"""Phase 3B-2a (lc0-split): the C++ self-play client's HTTP surface (cpp-httplib)
speaks the EXISTING fleet contract against a live in-process Python fleet_server.

Covers the two BINARY operations (wire-format parity is the risk):
  - cpp.fleet_get_network(base, sha)            GET  /get_network?sha=
  - cpp.fleet_upload_game(base, name, chunk, …) POST /upload_game (multipart)
plus cpp.fleet_next_game (job claim, body returned verbatim — C++ JSON parse is 3B-2b)
and the full client round-trip: next_game -> get_network -> play_game_chunk ->
upload_game lands a ccz chunk the Python server buffers and decode_chunk reads.
No torch on the client side of the wire; the C++ does the sockets (GIL released).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

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


def test_cpp_next_game_returns_job(server):
    base, rd = server
    train_continuous._publish(ChesskersScorer(), rd / "weights.pt")
    status, body = cpp.fleet_next_game(base)
    assert status == 200
    job = json.loads(body)
    assert job["type"] == "train" and job["bin_sha"]


def test_cpp_get_network_fetches_loadable_net(server, tmp_path):
    base, rd = server
    train_continuous._publish(ChesskersScorer(), rd / "weights.pt")
    bin_sha = hashlib.sha256((rd / "weights.bin").read_bytes()).hexdigest()

    status, net_bytes = cpp.fleet_get_network(base, bin_sha)
    assert status == 200
    assert hashlib.sha256(net_bytes).hexdigest() == bin_sha
    net_file = tmp_path / "fetched.bin"
    net_file.write_bytes(net_bytes)
    cpp.ChesskersNet(str(net_file))  # the fetched bytes load directly


def test_cpp_get_network_404(server):
    base, _rd = server
    status, _ = cpp.fleet_get_network(base, "0" * 64)
    assert status == 404


def test_cpp_upload_game_lands_in_buffer(server, tmp_path):
    base, rd = server
    train_continuous._publish(ChesskersScorer(), rd / "weights.pt")
    net = cpp.ChesskersNet(str(rd / "weights.bin"))
    chunk = cpp.play_game_chunk(cpp.parse_fen(SEED_FEN), net, n_sims=12, temperature=1.0,
                                temp_cutoff_plies=4, max_plies=20, dirichlet_alpha=0.3, seed=2)
    name = "300_0000000001.pkl"
    meta = b'{"machine": "cpp-client", "outcome": "draw"}'

    status, resp = cpp.fleet_upload_game(base, name, chunk, meta)
    assert status == 200 and resp == b"ok"
    # landed byte-identical, with its meta, and decodes as a ccz chunk
    landed = (rd / "buffer" / name).read_bytes()
    assert landed == chunk
    assert (rd / "buffer" / (name + ".meta")).read_bytes() == meta
    examples = decode_chunk(landed)
    assert examples and all(abs(sum(e.visit_distribution) - 1.0) < 1e-6 for e in examples)


def test_cpp_upload_rejects_bad_filename(server, tmp_path):
    base, rd = server
    train_continuous._publish(ChesskersScorer(), rd / "weights.pt")
    net = cpp.ChesskersNet(str(rd / "weights.bin"))
    chunk = cpp.play_game_chunk(cpp.parse_fen(SEED_FEN), net, n_sims=8, max_plies=12, seed=1)
    status, _ = cpp.fleet_upload_game(base, "../etc/passwd", chunk, b"")
    assert status == 400


def test_cpp_full_client_roundtrip(server, tmp_path):
    """The whole client step in C++: claim a job, fetch its net by sha, play a game,
    upload the chunk — the server buffers it and decode_chunk recovers the examples."""
    base, rd = server
    train_continuous._publish(ChesskersScorer(), rd / "weights.pt")

    _, job_body = cpp.fleet_next_game(base)
    job = json.loads(job_body)
    assert job["type"] == "train"

    _, net_bytes = cpp.fleet_get_network(base, job["bin_sha"])
    net_file = tmp_path / "net.bin"
    net_file.write_bytes(net_bytes)
    net = cpp.ChesskersNet(str(net_file))

    chunk = cpp.play_game_chunk(cpp.parse_fen(SEED_FEN), net, n_sims=12, temperature=1.0,
                                temp_cutoff_plies=4, max_plies=20, dirichlet_alpha=0.3, seed=5)
    name = "300_0000000007.pkl"
    status, resp = cpp.fleet_upload_game(base, name, chunk, b'{"machine":"cpp"}')
    assert status == 200 and resp == b"ok"

    landed = (rd / "buffer" / name).read_bytes()
    assert decode_chunk(landed)
    with urllib.request.urlopen(f"{base}/status", timeout=5) as r:
        assert json.loads(r.read())["games_ingested"] == 1
