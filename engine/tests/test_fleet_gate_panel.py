"""Wire-contract tests for the distributed keep-best gate panel (the two lc0 caveats).

Caveat 2 — the WHOLE opponent panel is fleet-offloadable, not just vs-best:
  - POST /next_game round-robins (opponent x seed x side) over the published panel as lc0
    `match` jobs (the legacy GET /next_game + /net/* serving were retired in Phase D; nets
    are fetched content-addressed via GET /get_network?sha=);
  - POST /match_result persists the opponent tag;
  - fleet_arena._RemotePool buckets client outcomes by (opp, seed, side) and preserves
    results for opponents the (sequential) arena hasn't reached yet.

These exercise the server + the arena's pool with stdlib HTTP only — no torch, no nets,
no self-play — so they're fast and deterministic. The actual game-playing path
(fleet_match.MatchRunner) reuses the arena's runner and is covered by the engine's MCTS tests.
"""
from __future__ import annotations

import itertools
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from chessckers_engine import fleet_server


# --- in-process server fixture -------------------------------------------------

@pytest.fixture
def server(tmp_path):
    """A real fleet_server bound to an ephemeral port, wired exactly as main() does."""
    rd = tmp_path / "run"
    (rd / "match_nets").mkdir(parents=True)
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


def _post(url: str, data: bytes):
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, r.read()


def _open_gate(rd, *, match_id, seeds, opponents):
    (rd / "match.json").write_text(json.dumps({
        "match_id": match_id, "seeds": seeds, "opponents": opponents,
        "arch": {"d_hidden": 8}, "params": {"sims": 1, "c_puct": 1.5,
                                             "dir_alpha": 0.5, "dir_eps": 0.15, "max_plies": 10},
    }))


# --- POST /next_game (the sole assignment path; GET was retired in Phase D) -----

def test_next_game_train_without_match(server):
    base, _rd = server
    _, body = _post(base + "/next_game", b"")
    assert json.loads(body)["type"] == "train"


def test_next_game_covers_every_opponent_seed_side(server):
    """Round-robin must enumerate the full (opponent x seed x side) product — the older
    champions are dispatched to clients as `match` jobs, not played only locally."""
    base, rd = server
    seeds = ["fenA", "fenB"]
    opps = ["best", "net-100", "net-200"]
    _open_gate(rd, match_id=42, seeds=seeds, opponents=opps)

    n_units = len(opps) * len(seeds) * 2
    seen = set()
    for _ in range(n_units * 2):  # two full sweeps
        _, body = _post(base + "/next_game", b"")
        a = json.loads(body)
        assert a["type"] == "match" and a["match_id"] == 42
        assert a["params"]["sims"] == 1  # gate params forwarded verbatim
        seen.add((a["opponent"], a["seed"], a["cand_white"]))
    assert seen == {(o, s, cw) for o in opps for s in seeds for cw in (True, False)}


def test_next_game_defaults_opponents_to_best(server):
    """A manifest without an `opponents` key (old-shape) still dispatches the vs-best gate."""
    base, rd = server
    (rd / "match.json").write_text(json.dumps({
        "match_id": 1, "seeds": ["s"], "arch": {}, "params": {"sims": 1},
    }))
    opps = {json.loads(_post(base + "/next_game", b"")[1])["opponent"] for _ in range(4)}
    assert opps == {"best"}


# --- /match_result -------------------------------------------------------------

def test_match_result_persists_opponent(server):
    base, rd = server
    _open_gate(rd, match_id=7, seeds=["s"], opponents=["best", "net-1"])

    status, _ = _post(base + "/match_result", json.dumps({
        "match_id": 7, "seed": "s", "opp": "net-1", "cand_white": True, "outcome": "white",
    }).encode())
    assert status == 200
    files = list((rd / "match_results").glob("7_*.json"))
    assert len(files) == 1
    rec = json.loads(files[0].read_text())
    assert rec == {"seed": "s", "cand_white": True, "opp": "net-1",
                   "outcome": "white", "match_id": 7}

    # A result for a rotated/closed gate is acked and dropped, never tallied.
    status, body = _post(base + "/match_result", json.dumps({
        "match_id": 999, "seed": "s", "opp": "best", "cand_white": False, "outcome": "draw",
    }).encode())
    assert status == 200 and body == b"stale"
    assert len(list((rd / "match_results").glob("*.json"))) == 1


# --- _RemotePool (the arena side) ----------------------------------------------

def test_remote_pool_buckets_by_opponent(tmp_path):
    """The arena plays opponents SEQUENTIALLY while clients supply all of them at once, so
    a drain triggered while consuming one opponent must not discard another's results."""
    from chessckers_engine.fleet_arena import _RemotePool

    rd = tmp_path / "match_results"
    rd.mkdir()
    mid = 5

    def write(n, opp, seed, cw, outcome):
        (rd / f"{mid}_{n}.json").write_text(json.dumps({
            "opp": opp, "seed": seed, "cand_white": cw, "outcome": outcome, "match_id": mid,
        }))

    write(0, "best", "s1", True, "white")
    write(1, "net-1", "s1", True, "black")   # different opponent, SAME seed+side
    write(2, "best", "s1", True, "draw")

    pool = _RemotePool(rd, mid)
    v_best, v_old = pool.view("best"), pool.view("net-1")

    # Consuming 'best' drains the whole dir (capturing net-1's result into the shared buf).
    got = sorted(filter(None, [v_best.pop("s1", True), v_best.pop("s1", True)]))
    assert got == ["draw", "white"]
    assert v_best.pop("s1", True) is None            # best exhausted

    # net-1's result survived the best-triggered drain — that's the shared-pool invariant.
    assert v_old.pop("s1", True) == "black"
    assert v_old.pop("s1", True) is None
    assert pool.popped == 3                           # summed across views


def test_remote_pool_unsupplied_unit_returns_none(tmp_path):
    from chessckers_engine.fleet_arena import _RemotePool
    rd = tmp_path / "match_results"
    rd.mkdir()
    pool = _RemotePool(rd, 1)
    assert pool.view("best").pop("missing", True) is None
    assert pool.popped == 0
