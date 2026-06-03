"""Fleet server — lc0-style training-server gateway.

Co-located with the continuous trainer (`train_continuous`), sharing its
`--run-dir` (the trainer writes `weights.pt` + `iter-async-*.pt` there and drains
`buffer/`). This process is the NETWORK FACE of that run-dir: self-play CLIENTS
(`fleet_client`, local over loopback + leena/volunteers over the LAN) pull the
current net and push finished games over plain HTTP. No new deps — stdlib
`http.server` only, so a volunteer box needs nothing but Python.

Endpoints:
  GET  /version          -> current net version string (changes once per trainer
                            ITERATION: tracks the newest iter-async-*.pt checkpoint,
                            falling back to weights.pt before the first checkpoint).
                            Clients poll this cheaply and only re-download on change
                            — same per-iteration cadence as the rsync sidecar it
                            replaces, now generalized to every client.
  GET  /weights          -> octet-stream of the freshest weights.pt + X-Version hdr.
  GET  /control          -> "RUN" or "STOP" (STOP once the trainer touches run/STOP).
  GET  /status           -> small JSON: version, weights present, buffer backlog, and
                            the fleet's live clients (any box whose request carried an
                            `X-Client-Id` within the last CLIENT_ACTIVE_WINDOW seconds).
  POST /game/<filename>  -> ingest one game artifact (`NNN_..pkl` or its `.pkl.meta`)
                            into buffer/ for the trainer to drain. pkl written
                            atomically; filename validated (no path traversal).

Keep-best gate distribution (lc0-style; active only while the arena has a gate open,
i.e. `match.json` is present in the run-dir):
  GET  /next_game        -> {"mode":"selfplay"} normally, else a match assignment
                            {"mode":"match", match_id, seed, cand_white, arch, params}.
                            Units (seed x side) are handed out round-robin; the arena
                            tolerates duplicates and plays whatever the fleet didn't.
  GET  /net/best,/net/cand -> octet-stream of best.pt / cand.pt (the two gate nets).
  POST /match_result     -> ingest one client-played gate outcome (JSON) into
                            match_results/ for the arena to tally. Outcomes whose
                            match_id != the open gate's are acked and dropped (stale).

Run (on the trainer host, same run-dir as train_continuous):

    python -m chessckers_engine.fleet_server --run-dir weights/run --port 8000
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chessckers_engine.runtime import setup_logging

log = logging.getLogger("chessckers_engine.fleet_server")

# Game artifacts are named `<worker_id:03d>_<game_id:010d>.pkl` (+ a `.pkl.meta`
# sidecar) by ReplayBuffer.append_game / the worker. worker_id may exceed 3 digits
# (volunteer bases), so allow 3+. Anchored — rejects any path traversal.
_NAME_RE = re.compile(r"^\d{3,}_\d{10}\.pkl(\.meta)?$")
MAX_UPLOAD_BYTES = 64 * 1024 * 1024  # a single game pkl is tens of KB; 64 MB is paranoia
CLIENT_ACTIVE_WINDOW = 120  # s — a client last-seen within this counts as "active" in /status


def _net_path(run_dir: Path) -> Path:
    """The net file clients should pull: the gated champion `best.pt` if the
    arena (fleet_arena) is maintaining one, else the trainer's raw `weights.pt`."""
    best = run_dir / "best.pt"
    return best if best.exists() else run_dir / "weights.pt"


def _version(run_dir: Path) -> str:
    """Net version the clients key off. When keep-best gating is active, tracks
    `best.pt`'s promotion mtime — clients refresh once per PROMOTION. With no
    arena running, falls back to the newest *checkpoint* (per-iteration cadence),
    then to weights.pt before the first checkpoint, so behaviour is unchanged when
    fleet_arena isn't deployed."""
    best = run_dir / "best.pt"
    if best.exists():
        return f"best:{int(best.stat().st_mtime)}"
    ckpts = list(run_dir.glob("iter-async-*.pt"))
    if ckpts:
        newest = max(int(p.stat().st_mtime) for p in ckpts)
        return f"ckpt:{newest}"
    w = run_dir / "weights.pt"
    if w.exists():
        return f"init:{int(w.stat().st_mtime)}"
    return "none"


def _read_match(run_dir: Path) -> dict | None:
    """The open-gate manifest the arena publishes (candidate vs best). Absent
    between gates — clients then get self-play. Arena writes it atomically, so a
    read never sees a half-file."""
    try:
        return json.loads((run_dir / "match.json").read_text())
    except (OSError, ValueError):
        return None


class _Handler(BaseHTTPRequestHandler):
    server_version = "ChesskersFleet/1.0"
    run_dir: Path  # set on the server instance below; bound per-request via self.server

    def _run_dir(self) -> Path:
        return self.server.run_dir  # type: ignore[attr-defined]

    def _note_client(self) -> None:
        """Stamp the calling box's last-seen time for the /status fleet view. Best-effort
        and header-only — a client that sends no X-Client-Id is simply invisible here."""
        cid = self.headers.get("X-Client-Id")
        if not cid:
            return
        with self.server.clients_lock:  # type: ignore[attr-defined]
            self.server.clients[cid[:64]] = time.time()  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *a) -> None:  # silence default stderr access log
        log.debug("%s - " + fmt, self.address_string(), *a)

    def _send(self, code: int, body: bytes = b"", ctype: str = "text/plain",
              extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self) -> None:
        self._note_client()
        rd = self._run_dir()
        if self.path == "/version":
            self._send(200, _version(rd).encode())
        elif self.path == "/control":
            self._send(200, b"STOP" if (rd / "STOP").exists() else b"RUN")
        elif self.path == "/weights":
            try:
                data = _net_path(rd).read_bytes()
            except OSError:
                self._send(404, b"no weights yet")
                return
            self._send(200, data, "application/octet-stream",
                       {"X-Version": _version(rd)})
        elif self.path == "/status":
            backlog = sum(1 for _ in (rd / "buffer").glob("*.pkl")) if (rd / "buffer").exists() else 0
            m = _read_match(rd)
            now = time.time()
            with self.server.clients_lock:  # type: ignore[attr-defined]
                clients = {cid: round(now - ts, 1)
                           for cid, ts in self.server.clients.items()  # type: ignore[attr-defined]
                           if now - ts <= CLIENT_ACTIVE_WINDOW}
            body = json.dumps({
                "version": _version(rd),
                "weights": (rd / "weights.pt").exists(),
                "best": (rd / "best.pt").exists(),
                "buffer_backlog": backlog,
                "control": "STOP" if (rd / "STOP").exists() else "RUN",
                "gate_open": bool(m),
                "clients_active": len(clients),
                "clients": clients,  # {id: seconds-since-last-seen}, active window only
            }).encode()
            self._send(200, body, "application/json")
        elif self.path == "/next_game":
            m = _read_match(rd)
            seeds = (m or {}).get("seeds") or []
            if not m or not seeds:
                self._send(200, b'{"mode": "selfplay"}', "application/json")
                return
            units = [(s, cw) for s in seeds for cw in (True, False)]
            seed, cand_white = units[next(self.server.match_cursor) % len(units)]  # type: ignore[attr-defined]
            self._send(200, json.dumps({
                "mode": "match", "match_id": m["match_id"], "seed": seed,
                "cand_white": cand_white, "arch": m["arch"], "params": m["params"],
            }).encode(), "application/json")
        elif self.path in ("/net/best", "/net/cand"):
            fname = "best.pt" if self.path.endswith("best") else "cand.pt"
            try:
                data = (rd / fname).read_bytes()
            except OSError:
                self._send(404, b"no net")
                return
            self._send(200, data, "application/octet-stream")
        else:
            self._send(404, b"not found")

    def do_POST(self) -> None:
        self._note_client()
        if self.path == "/match_result":
            self._match_result()
            return
        if not self.path.startswith("/game/"):
            self._send(404, b"not found")
            return
        name = self.path[len("/game/"):]
        if not _NAME_RE.match(name):
            self._send(400, b"bad filename")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, b"bad length")
            return
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self._send(400, b"bad length")
            return
        data = self.rfile.read(length)
        buf = self._run_dir() / "buffer"
        buf.mkdir(parents=True, exist_ok=True)
        target = buf / name
        # Atomic for the .pkl so the trainer's drain never sees a half-written game
        # (it globs *.pkl and pickle.loads each). The .meta is a small single write.
        tmp = target.with_name(target.name + ".part")
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, target)
        except OSError as e:
            log.warning("write %s failed: %s", name, e)
            self._send(500, b"write failed")
            return
        self._send(200, b"ok")

    def _match_result(self) -> None:
        """Ingest one client-played gate outcome into match_results/ for the arena.
        Stale results (a different/closed match) are acked and dropped so a client
        finishing a unit after the gate rotated doesn't pollute the next gate."""
        rd = self._run_dir()
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, b"bad length")
            return
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self._send(400, b"bad length")
            return
        try:
            r = json.loads(self.rfile.read(length))
        except ValueError:
            self._send(400, b"bad json")
            return
        m = _read_match(rd)
        if not m or r.get("match_id") != m.get("match_id"):
            self._send(200, b"stale")  # gate closed/rotated — ack so the client drops it
            return
        if r.get("outcome") not in ("white", "black", "draw"):
            self._send(400, b"bad outcome")
            return
        mrd = rd / "match_results"
        mrd.mkdir(parents=True, exist_ok=True)
        n = next(self.server.result_counter)  # type: ignore[attr-defined]
        target = mrd / f"{m['match_id']}_{n}.json"
        body = json.dumps({"seed": r.get("seed"), "cand_white": bool(r.get("cand_white")),
                           "outcome": r["outcome"], "match_id": m["match_id"]}).encode()
        tmp = target.with_name(target.name + ".part")
        try:
            with open(tmp, "wb") as f:
                f.write(body)
            os.replace(tmp, target)
        except OSError as e:
            log.warning("match_result write failed: %s", e)
            self._send(500, b"write failed")
            return
        self._send(200, b"ok")


def main() -> int:
    setup_logging()
    p = argparse.ArgumentParser(description="Fleet server: distribute net + ingest games (lc0-style).")
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Trainer's run-dir (shared FS): serves weights.pt, ingests into buffer/.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    (run_dir / "buffer").mkdir(parents=True, exist_ok=True)

    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    httpd.run_dir = run_dir  # type: ignore[attr-defined]
    httpd.match_cursor = itertools.count()    # round-robins gate units in /next_game
    httpd.result_counter = itertools.count()  # unique filenames for /match_result writes
    httpd.clients = {}                        # X-Client-Id -> last-seen epoch (fleet liveness)
    httpd.clients_lock = threading.Lock()     # guards clients across handler threads
    httpd.daemon_threads = True
    log.info("fleet server up on %s:%d (run-dir=%s, version=%s)",
             args.host, args.port, run_dir, _version(run_dir))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("fleet server stopping")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
