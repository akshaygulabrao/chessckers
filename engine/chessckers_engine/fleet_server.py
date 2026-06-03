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
  GET  /status           -> small JSON: version, weights present, buffer backlog.
  POST /game/<filename>  -> ingest one game artifact (`NNN_..pkl` or its `.pkl.meta`)
                            into buffer/ for the trainer to drain. pkl written
                            atomically; filename validated (no path traversal).

Run (on the trainer host, same run-dir as train_continuous):

    python -m chessckers_engine.fleet_server --run-dir weights/run --port 8000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chessckers_engine.runtime import setup_logging

log = logging.getLogger("chessckers_engine.fleet_server")

# Game artifacts are named `<worker_id:03d>_<game_id:010d>.pkl` (+ a `.pkl.meta`
# sidecar) by ReplayBuffer.append_game / the worker. worker_id may exceed 3 digits
# (volunteer bases), so allow 3+. Anchored — rejects any path traversal.
_NAME_RE = re.compile(r"^\d{3,}_\d{10}\.pkl(\.meta)?$")
MAX_UPLOAD_BYTES = 64 * 1024 * 1024  # a single game pkl is tens of KB; 64 MB is paranoia


def _version(run_dir: Path) -> str:
    """Net version the clients key off. Tracks the newest *checkpoint* so clients
    refresh once per trainer iteration (not on every ~45s weights.pt republish);
    falls back to weights.pt before the first checkpoint exists so the initial net
    is still distributed promptly."""
    ckpts = list(run_dir.glob("iter-async-*.pt"))
    if ckpts:
        newest = max(int(p.stat().st_mtime) for p in ckpts)
        return f"ckpt:{newest}"
    w = run_dir / "weights.pt"
    if w.exists():
        return f"init:{int(w.stat().st_mtime)}"
    return "none"


class _Handler(BaseHTTPRequestHandler):
    server_version = "ChesskersFleet/1.0"
    run_dir: Path  # set on the server instance below; bound per-request via self.server

    def _run_dir(self) -> Path:
        return self.server.run_dir  # type: ignore[attr-defined]

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
        rd = self._run_dir()
        if self.path == "/version":
            self._send(200, _version(rd).encode())
        elif self.path == "/control":
            self._send(200, b"STOP" if (rd / "STOP").exists() else b"RUN")
        elif self.path == "/weights":
            w = rd / "weights.pt"
            try:
                data = w.read_bytes()
            except OSError:
                self._send(404, b"no weights yet")
                return
            self._send(200, data, "application/octet-stream",
                       {"X-Version": _version(rd)})
        elif self.path == "/status":
            backlog = sum(1 for _ in (rd / "buffer").glob("*.pkl")) if (rd / "buffer").exists() else 0
            body = json.dumps({
                "version": _version(rd),
                "weights": (rd / "weights.pt").exists(),
                "buffer_backlog": backlog,
                "control": "STOP" if (rd / "STOP").exists() else "RUN",
            }).encode()
            self._send(200, body, "application/json")
        else:
            self._send(404, b"not found")

    def do_POST(self) -> None:
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
