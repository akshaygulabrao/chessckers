"""Wire-contract tests for the lc0-canonical fleet endpoints (Phase A + B).

These cover the additive lc0-shaped surface laid over the existing fleet_server:
  - GET  /get_network?sha=  content-addressed net fetch (+ X-Network-Sha on /weights, net_sha in /status)
  - POST /next_game         lc0 job JSON ({"type":"train"|"match","sha":...,"params":...})
  - POST /upload_game       multipart/form-data game upload -> buffer/ (stdlib parser; py3.13 has no cgi)
  - _parse_multipart        the hand-rolled parser, incl. binary payloads ending in CRLF

Phase B (content-addressed net sync, client side):
  - GET  /control           advertises X-Network-Sha (the current net's content address)
  - GET  /status            reports each client's running net sha (X-Client-Net)
  - fleet_client._pull_net_by_sha  fetches by sha, no-ops when unchanged/empty

Phase C (gzipped-chunk records + client upload migration):
  - fleet_client._build_multipart  ↔ fleet_server._parse_multipart  agree on the wire
  - fleet_client._upload_games     posts each game to /upload_game (multipart), not /game

Phase D (gate assignment folded into the client; the legacy GET /next_game + /net/* serving
were RETIRED — POST /next_game is now the sole assignment path):
  - fleet_client._play_gate_job    plays one `match` job: fetch cand+opp nets by sha
    (/get_network), play (lazy MatchRunner — stubbed here), POST /match_result
  - run-dir/PAUSE                  sentinel the client touches so its owned workers idle mid-gate

and assert the LEGACY self-play endpoints (POST /game, GET /version) still work — the lc0 mirror
must not break the existing fleet_client self-play path. Stdlib HTTP only: fast, no torch/nets.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from chessckers_engine import fleet_client, fleet_server


# --- in-process server fixture (same wiring as fleet_server.main) ---------------

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


def _get(url: str, headers: dict | None = None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, r.read(), dict(r.headers)


def _post(url: str, data: bytes, ctype: str = "application/octet-stream"):
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, r.read()


def _multipart(parts):
    """parts: list of (name, filename_or_None, bytes) -> (content_type, body)."""
    boundary = "----ccWireBoundaryZ9"
    out = b""
    for name, filename, data in parts:
        disp = f'Content-Disposition: form-data; name="{name}"'
        if filename is not None:
            disp += f'; filename="{filename}"'
        out += f"--{boundary}\r\n".encode() + disp.encode() + b"\r\n\r\n" + data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return f"multipart/form-data; boundary={boundary}", out


# --- the parser itself (binary-safety) -----------------------------------------

def test_parse_multipart_preserves_binary_payload():
    """A payload ending in CRLF must round-trip byte-for-byte (the strip() bug class)."""
    blob = b"\x00\x01\x02pickle-bytes\r\n"  # ends in CRLF on purpose
    ctype, body = _multipart([("filename", None, b"300_0000000001.pkl"),
                              ("trainingdata", "g.pkl", blob)])
    parts = fleet_server._parse_multipart(ctype, body)
    assert parts["trainingdata"][1] == blob
    assert parts["filename"][1] == b"300_0000000001.pkl"


# --- GET /get_network ----------------------------------------------------------

def test_get_network_serves_by_sha_and_404s(server):
    base, rd = server
    blob = b"NET-WEIGHTS-v0"
    (rd / "weights.pt").write_bytes(blob)
    sha = hashlib.sha256(blob).hexdigest()

    status, body, _ = _get(f"{base}/get_network?sha={sha}")
    assert status == 200 and body == blob

    # /weights and /status advertise that same sha, so a client knows what to ask for.
    _, _, whdrs = _get(f"{base}/weights")
    assert whdrs.get("X-Network-Sha") == sha
    _, sbody, _ = _get(f"{base}/status")
    assert json.loads(sbody)["net_sha"] == sha

    with pytest.raises(urllib.error.HTTPError) as e:
        _get(f"{base}/get_network?sha=" + "0" * 64)
    assert e.value.code == 404

    with pytest.raises(urllib.error.HTTPError) as e:  # missing sha -> 404
        _get(f"{base}/get_network")
    assert e.value.code == 404


# --- POST /next_game (lc0 shape) -----------------------------------------------

def test_next_game_post_train_job(server):
    base, rd = server
    (rd / "weights.pt").write_bytes(b"W")
    (rd / "selfplay.json").write_text(json.dumps({"sims": 7, "c_puct": 1.5}))
    sha = hashlib.sha256(b"W").hexdigest()

    status, body = _post(f"{base}/next_game", b"user=x")  # body (client identity) ignored
    job = json.loads(body)
    assert status == 200
    assert job["type"] == "train"
    assert job["sha"] == sha
    assert job["params"]["sims"] == 7  # selfplay.json forwarded in the job


def test_next_game_post_match_job(server):
    base, rd = server
    (rd / "cand.pt").write_bytes(b"CAND")
    (rd / "best.pt").write_bytes(b"BEST")
    (rd / "match_nets" / "net-100.pt").write_bytes(b"CHAMP100")
    (rd / "match.json").write_text(json.dumps({
        "match_id": 9, "seeds": ["s"], "opponents": ["best", "net-100"],
        "arch": {"d_hidden": 8}, "params": {"sims": 1},
    }))
    cand_sha = hashlib.sha256(b"CAND").hexdigest()
    opp_sha = {"best": hashlib.sha256(b"BEST").hexdigest(),
               "net-100": hashlib.sha256(b"CHAMP100").hexdigest()}

    seen = set()
    for _ in range(8):  # full (2 opp x 1 seed x 2 side) product, twice
        _, body = _post(f"{base}/next_game", b"")
        job = json.loads(body)
        assert job["type"] == "match" and job["match_id"] == 9
        assert job["candidate_sha"] == cand_sha and job["sha"] == cand_sha
        assert job["opponent_sha"] == opp_sha[job["opponent"]]  # content-addressed opponent
        assert job["params"]["sims"] == 1
        seen.add((job["opponent"], job["seed"], job["cand_white"]))
    assert seen == {(o, "s", cw) for o in ("best", "net-100") for cw in (True, False)}


# --- POST /upload_game ---------------------------------------------------------

def test_upload_game_lands_pkl_and_meta(server):
    base, rd = server
    pkl = b"\x80\x04pickled-game\r\n"   # binary, CRLF-terminated
    meta = b'{"worker": 300}'
    name = "300_0000000042.pkl"
    ctype, body = _multipart([("filename", None, name.encode()),
                              ("trainingdata", "g.pkl", pkl),
                              ("meta", "g.pkl.meta", meta)])
    status, resp = _post(f"{base}/upload_game", body, ctype)
    assert status == 200 and resp == b"ok"

    assert (rd / "buffer" / name).read_bytes() == pkl
    assert (rd / "buffer" / (name + ".meta")).read_bytes() == meta
    # counted as one ingested game in /status throughput
    _, sbody, _ = _get(f"{base}/status")
    assert json.loads(sbody)["games_ingested"] == 1


def test_upload_game_rejects_bad_filename(server):
    base, _rd = server
    ctype, body = _multipart([("filename", None, b"../etc/passwd"),
                              ("trainingdata", "x", b"data")])
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(f"{base}/upload_game", body, ctype)
    assert e.value.code == 400


# --- non-breaking: legacy endpoints still work ---------------------------------

def test_legacy_game_upload_still_works(server):
    base, rd = server
    name = "300_0000000007.pkl"
    status, resp = _post(f"{base}/game/{name}", b"raw-pickle-bytes")
    assert status == 200 and resp == b"ok"
    assert (rd / "buffer" / name).read_bytes() == b"raw-pickle-bytes"
    _, sbody, _ = _get(f"{base}/status")
    assert json.loads(sbody)["games_ingested"] == 1


def test_version_endpoint_unchanged(server):
    base, rd = server
    (rd / "weights.pt").write_bytes(b"W")
    status, body, _ = _get(f"{base}/version")
    assert status == 200 and body.decode().startswith("init:")


# --- Phase B: content-addressed net sync (client side) -------------------------

def test_control_advertises_net_sha(server):
    """GET /control carries X-Network-Sha so a client syncs the net on its heartbeat tick."""
    base, rd = server
    # before any net exists: header present but empty (nothing to pull)
    _, body, hdrs = _get(f"{base}/control")
    assert body == b"RUN" and hdrs.get("X-Network-Sha") == ""
    # once weights.pt lands: /control advertises its content address (same sha /weights serves)
    (rd / "weights.pt").write_bytes(b"NET-ABC")
    sha = hashlib.sha256(b"NET-ABC").hexdigest()
    _, _, hdrs = _get(f"{base}/control")
    assert hdrs.get("X-Network-Sha") == sha
    _, _, whdrs = _get(f"{base}/weights")
    assert whdrs.get("X-Network-Sha") == sha
    # best.pt is the preferred servable net, so the advertised sha tracks it
    (rd / "best.pt").write_bytes(b"BEST-NET")
    best_sha = hashlib.sha256(b"BEST-NET").hexdigest()
    _, _, hdrs = _get(f"{base}/control")
    assert hdrs.get("X-Network-Sha") == best_sha


def test_status_reports_client_net(server):
    """A heartbeat carrying X-Client-Net surfaces that box's running net sha in /status."""
    base, rd = server
    sha = hashlib.sha256(b"W").hexdigest()
    _get(f"{base}/control", {"X-Client-Id": "leena", "X-Client-Net": sha, "X-Client-Workers": "up"})
    _, sbody, _ = _get(f"{base}/status")
    client = json.loads(sbody)["clients"]["leena"]
    assert client["net"] == sha and client["workers"] == "up"


def test_client_pull_net_by_sha(server, tmp_path):
    """fleet_client._pull_net_by_sha: fetch-by-sha materializes weights.pt; no-op on
    unchanged sha and on empty want (the content-addressed sync the client now runs)."""
    base, rd = server
    blob = b"NET-WEIGHTS-xyz"
    (rd / "weights.pt").write_bytes(blob)
    sha = hashlib.sha256(blob).hexdigest()
    w = tmp_path / "client" / "weights.pt"
    w.parent.mkdir()

    # empty want -> no-op, nothing written
    assert fleet_client._pull_net_by_sha(base, w, "", "", 5) == ""
    assert not w.exists()

    # fetch by sha -> materializes the net at weights.pt, returns the sha now on disk
    assert fleet_client._pull_net_by_sha(base, w, sha, "", 5) == sha
    assert w.read_bytes() == blob

    # unchanged sha -> no-op: returns the same sha and does NOT re-fetch/rewrite
    w.write_bytes(b"SENTINEL")  # if the no-op path re-fetched, this would be overwritten
    assert fleet_client._pull_net_by_sha(base, w, sha, sha, 5) == sha
    assert w.read_bytes() == b"SENTINEL"


# --- Phase C: client uploads via multipart /upload_game ------------------------

def test_client_multipart_matches_server_parser():
    """The client's multipart builder and the server's hand-rolled parser agree on
    the wire — incl. a binary payload ending in CRLF (the strip() trap) and a
    value-only part with no filename."""
    blob = b"\x1f\x8bbinary\x00chunk\r\n"  # gzip-magic + binary, CRLF-terminated
    ctype, body = fleet_client._build_multipart([
        ("filename", None, b"300_0000000009.pkl"),
        ("trainingdata", "g.pkl", blob),
        ("meta", "g.pkl.meta", b'{"machine":"local"}'),
    ])
    parts = fleet_server._parse_multipart(ctype, body)
    assert parts["filename"][1] == b"300_0000000009.pkl"
    assert parts["trainingdata"][1] == blob          # byte-exact, CRLF preserved
    assert parts["meta"][1] == b'{"machine":"local"}'


def test_client_uploads_via_multipart_endpoint(server, tmp_path):
    """fleet_client._upload_games posts each game to /upload_game as ONE multipart
    request and deletes it locally; the bytes (a ccz chunk, opaque to the client)
    land in the server's buffer/ with the .meta, counted once."""
    base, rd = server
    buf = tmp_path / "client_buf"
    buf.mkdir()
    name = "300_0000000005.pkl"
    game_bytes = b"\x1f\x8b" + b"opaque-ccz-chunk-bytes\r\n"  # client never parses these
    meta_bytes = b'{"machine": "leena", "outcome": "white"}'
    (buf / name).write_bytes(game_bytes)
    (buf / (name + ".meta")).write_bytes(meta_bytes)

    n = fleet_client._upload_games(base, buf, min_age=0.0, timeout=5)
    assert n == 1
    # landed server-side byte-identical, with its meta, counted once
    assert (rd / "buffer" / name).read_bytes() == game_bytes
    assert (rd / "buffer" / (name + ".meta")).read_bytes() == meta_bytes
    _, sbody, _ = _get(f"{base}/status")
    assert json.loads(sbody)["games_ingested"] == 1
    # consumed locally — uploaded exactly once
    assert not (buf / name).exists()
    assert not (buf / (name + ".meta")).exists()


# --- Phase D: client plays gate jobs over POST /next_game ----------------------

def test_client_plays_gate_job_over_post(server, tmp_path, monkeypatch):
    """fleet_client._play_gate_job: given an open gate's `match` job, the client fetches the
    candidate + opponent nets by content address (/get_network?sha=) and POSTs /match_result.
    The heavy player is stubbed so the wire test stays torch-free; we assert it received the
    right nets (content-addressed, cached on disk) and that the outcome landed for the arena."""
    import types

    base, rd = server
    (rd / "cand.pt").write_bytes(b"CANDNET")
    (rd / "best.pt").write_bytes(b"BESTNET")
    (rd / "match.json").write_text(json.dumps({
        "match_id": 11, "seeds": ["fenX"], "opponents": ["best"],
        "arch": {"d_hidden": 8}, "params": {"sims": 1, "c_puct": 1.5,
                                            "dir_alpha": 0.5, "dir_eps": 0.15, "max_plies": 10},
    }))

    # the match job the client would receive from POST /next_game (nets identified by sha)
    _, body = _post(f"{base}/next_game", b"")
    job = json.loads(body)
    assert job["type"] == "match" and job["candidate_sha"] and job["opponent_sha"]

    # stub the heavy player (no torch); capture the nets it's handed
    seen = {}

    class _FakeRunner:
        def __init__(self, cache_dir, device="cpu"):
            seen["device"] = device

        def play(self, j, cand_path, opp_path):
            seen["job_mid"] = j["match_id"]
            seen["cand"] = cand_path.read_bytes()
            seen["opp"] = opp_path.read_bytes()
            return "white"

    fake = types.ModuleType("chessckers_engine.fleet_match")
    fake.MatchRunner = _FakeRunner
    monkeypatch.setitem(sys.modules, "chessckers_engine.fleet_match", fake)

    gate_dir = tmp_path / "_gate"
    runner, disabled = fleet_client._play_gate_job(base, job, gate_dir, None, "cpu", 5)
    assert disabled is False and runner is not None

    # fetched the right nets by content address, and cached them sha-named on disk
    assert seen["cand"] == b"CANDNET" and seen["opp"] == b"BESTNET" and seen["job_mid"] == 11
    assert (gate_dir / f"{job['candidate_sha']}.pt").read_bytes() == b"CANDNET"
    # the outcome landed (tagged with the opponent) for the arena to tally
    files = list((rd / "match_results").glob("11_*.json"))
    assert len(files) == 1
    rec = json.loads(files[0].read_text())
    assert rec["outcome"] == "white" and rec["match_id"] == 11 and rec["opp"] == job["opponent"]


def test_play_gate_job_disables_when_engine_missing(server, tmp_path, monkeypatch):
    """A box without torch/the native ext disables gating (returns disabled=True) instead of
    crashing, so a bare volunteer stays self-play-only — the lazy-import guard."""
    base, rd = server
    (rd / "cand.pt").write_bytes(b"C")
    (rd / "best.pt").write_bytes(b"B")
    (rd / "match.json").write_text(json.dumps({
        "match_id": 1, "seeds": ["s"], "opponents": ["best"], "arch": {},
        "params": {"sims": 1, "c_puct": 1.5, "dir_alpha": 0.5, "dir_eps": 0.15, "max_plies": 5},
    }))
    _, body = _post(f"{base}/next_game", b"")
    job = json.loads(body)

    # a None entry in sys.modules makes `from ... import MatchRunner` raise ImportError
    monkeypatch.setitem(sys.modules, "chessckers_engine.fleet_match", None)
    runner, disabled = fleet_client._play_gate_job(base, job, tmp_path / "_gate", None, "cpu", 5)
    assert runner is None and disabled is True


def test_worker_pause_sentinel_semantics(tmp_path):
    """The self-play worker idles while run-dir/PAUSE exists (the client's gate-pause). PAUSE is
    derived from the STOP path the worker already receives, so it sits in the same run-dir."""
    from chessckers_engine.selfplay_worker_async import _pause_requested
    pause = tmp_path / "PAUSE"
    assert _pause_requested(None) is False
    assert _pause_requested(pause) is False
    pause.touch()
    assert _pause_requested(pause) is True
    # contract: the client touches run-dir/PAUSE; the worker derives it as stop_path.with_name("PAUSE")
    assert (tmp_path / "STOP").with_name("PAUSE") == pause
