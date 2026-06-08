"""Phase 4a (lc0-split): the gate path publishes C++-loadable .bin twins for the
candidate + every opponent net, and the match job carries their shas — so a native
client can fetch BOTH gate nets by content address and play the gate game (the C++
match play itself is 4b). The 3B-1 .bin publishing, extended from the train net to
the gate nets; additive (Python match runner uses the .pt shas, untouched).

  - fleet_arena._export_bin / _publish_gate_bin  write the .bin beside a gate .pt
  - POST /next_game match job carries candidate_bin_sha + opponent_bin_sha, both
    fetchable via GET /get_network?sha=
"""
from __future__ import annotations

import hashlib
import itertools
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
import torch

from chessckers_engine import fleet_arena, fleet_server
from chessckers_engine.model import build_model

cpp = pytest.importorskip("chessckers_cpp")


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


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read()


def _post(url, data):
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, r.read()


def _save_net(path, arch):
    m = build_model(**arch)
    m.eval()
    torch.save(m.state_dict(), path)


def test_export_bin_is_loadable(tmp_path):
    arch = {"version": "v1"}
    pt = tmp_path / "x.pt"
    _save_net(pt, arch)
    binp = tmp_path / "x.bin"
    fleet_arena._export_bin(pt, arch, binp)
    assert binp.exists()
    cpp.ChesskersNet(str(binp))  # loads directly into the C++ net


def test_publish_gate_bin_is_best_effort(tmp_path):
    """A bad/missing source never raises (would otherwise block the gate)."""
    fleet_arena._publish_gate_bin(tmp_path / "missing.pt", {"version": "v1"})
    assert not (tmp_path / "missing.bin").exists()


def test_match_job_carries_bin_shas(server):
    base, rd = server
    arch = {"version": "v1"}
    for name in ("cand.pt", "best.pt"):
        _save_net(rd / name, arch)
        fleet_arena._publish_gate_bin(rd / name, arch)
    _save_net(rd / "match_nets" / "net-100.pt", arch)
    fleet_arena._publish_gate_bin(rd / "match_nets" / "net-100.pt", arch)
    (rd / "match.json").write_text(json.dumps({
        "match_id": 5, "seeds": ["s"], "opponents": ["best", "net-100"],
        "arch": arch, "params": {"sims": 1},
    }))
    cand_bin_sha = hashlib.sha256((rd / "cand.bin").read_bytes()).hexdigest()
    opp_bin_sha = {
        "best": hashlib.sha256((rd / "best.bin").read_bytes()).hexdigest(),
        "net-100": hashlib.sha256((rd / "match_nets" / "net-100.bin").read_bytes()).hexdigest(),
    }

    seen = set()
    for _ in range(8):  # full opponent x side product, twice
        _, body = _post(f"{base}/next_game", b"")
        job = json.loads(body)
        assert job["type"] == "match"
        assert job["candidate_bin_sha"] == cand_bin_sha
        assert job["opponent_bin_sha"] == opp_bin_sha[job["opponent"]]
        # the C++ client can fetch the opponent net by that sha and load it
        st, net_bytes = _get(f"{base}/get_network?sha={job['opponent_bin_sha']}")
        assert st == 200 and hashlib.sha256(net_bytes).hexdigest() == job["opponent_bin_sha"]
        seen.add(job["opponent"])
    assert seen == {"best", "net-100"}


def test_match_job_bin_sha_empty_without_twin(server):
    """No .bin twins -> *_bin_sha is '' (additive: the .pt match path still works)."""
    base, rd = server
    (rd / "cand.pt").write_bytes(b"C")
    (rd / "best.pt").write_bytes(b"B")
    (rd / "match.json").write_text(json.dumps({
        "match_id": 1, "seeds": ["s"], "opponents": ["best"], "arch": {}, "params": {"sims": 1},
    }))
    _, body = _post(f"{base}/next_game", b"")
    job = json.loads(body)
    assert job["candidate_bin_sha"] == "" and job["opponent_bin_sha"] == ""
    assert job["candidate_sha"] == hashlib.sha256(b"C").hexdigest()  # .pt path intact
