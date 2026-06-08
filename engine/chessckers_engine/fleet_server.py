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
  GET  /control          -> "RUN" or "STOP" (STOP once the trainer touches run/STOP),
                            + an `X-Network-Sha` header = the current net's content
                            address, so a client syncs the net on the heartbeat tick it
                            already makes (content-addressed sync; fetch via /get_network).
  GET  /client-version   -> the trainer host's own git sha (resolved at server start).
                            Self-updating clients (fleet_client --update-cmd) compare it
                            to the sha they booted on and, on drift, pull + rebuild the
                            native ext + re-exec — closing the stale-.so failure class.
  GET  /selfplay         -> JSON of the canonical self-play params (run_dir/selfplay.json:
                            sims, c_puct, temperature, dirichlet_*, max_plies), or `{}`
                            if none published. Clients mirror it into their own run-dir
                            and the workers live-apply it at the next game boundary — so
                            every box self-plays with the SAME, operator-tunable params
                            (no more leena/local sims drift; anneal by editing the file).
  GET  /status           -> small JSON: version, weights present, buffer backlog, fleet
                            throughput (games ingested this run, promotions, best_elo,
                            open match_id), and the fleet's live clients with their code
                            version, worker-subprocess liveness, and the net sha they
                            report running (X-Client-Net — so fleet net-consistency is
                            visible: any box whose request carried an `X-Client-Id` within
                            the last CLIENT_ACTIVE_WINDOW seconds; workers=down is a zombie).
  POST /game/<filename>  -> ingest one game artifact (`NNN_..pkl` or its `.pkl.meta`)
                            into buffer/ for the trainer to drain. pkl written
                            atomically; filename validated (no path traversal).

lc0-canonical wire (mirrors lc0's server/client vocabulary; the self-play legacy
endpoints — POST /game, GET /version+/weights — stay for non-breaking):
  GET  /get_network?sha= -> content-addressed net fetch: the servable net (best/weights/
                            cand/opponent) whose sha256 == <sha>, else 404. /weights and
                            /status also carry the current net's sha (X-Network-Sha header /
                            net_sha field) so a client knows what to ask for. This is also how
                            a gate client fetches the candidate + each opponent net (by the
                            candidate_sha/opponent_sha the match job carries).
  POST /next_game        -> the SOLE job-assignment path (the legacy GET /next_game + /net/*
                            serving endpoints were retired in Phase D). lc0-shaped job JSON:
                            {"type":"train","sha":<net sha>,"params":...} normally, or, while a
                            gate is open (match.json present), a {"type":"match",match_id,
                            candidate_sha,opponent,opponent_sha,seed,cand_white,arch,params}
                            unit handed out round-robin over the whole panel (candidate vs the
                            last N champions). The arena tolerates duplicates and plays whatever
                            the fleet didn't. Request body (client identity) accepted and ignored.
  POST /upload_game      -> multipart/form-data game upload (parts: filename, trainingdata = a
                            gzipped-JSON `ccz` chunk since Phase C, optional meta). Lands into
                            buffer/ exactly like POST /game; a byte pipe (never parses the payload).
  POST /match_result     -> ingest one client-played gate outcome (JSON) into match_results/ for
                            the arena to tally. Outcomes whose match_id != the open gate's are
                            acked and dropped (stale).

Run (on the trainer host, same run-dir as train_continuous):

    python -m chessckers_engine.fleet_server --run-dir weights/run --port 8000
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from chessckers_engine.runtime import setup_logging

log = logging.getLogger("chessckers_engine.fleet_server")

# Game artifacts are named `<worker_id:03d>_<game_id:010d>.pkl` (+ a `.pkl.meta`
# sidecar) by ReplayBuffer.append_game / the worker. worker_id may exceed 3 digits
# (volunteer bases), so allow 3+. Anchored — rejects any path traversal.
_NAME_RE = re.compile(r"^\d{3,}_\d{10}\.pkl(\.meta)?$")
MAX_UPLOAD_BYTES = 64 * 1024 * 1024  # a single game pkl is tens of KB; 64 MB is paranoia
CLIENT_ACTIVE_WINDOW = 120  # s — a client last-seen within this counts as "active" in /status


def _code_version() -> str:
    """Short git sha of the server's own code, resolved once at startup and served at
    /client-version so self-updating clients can detect when they're behind the trainer
    host's code (the stale-.so class of silent failure). 'unknown' off a git checkout."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 — version is diagnostics-only; never block startup on it
        return "unknown"


def _net_path(run_dir: Path) -> Path:
    """The net file clients should pull: the gated champion `best.pt` if the
    arena (fleet_arena) is maintaining one, else the trainer's raw `weights.pt`."""
    best = run_dir / "best.pt"
    return best if best.exists() else run_dir / "weights.pt"


def _net_bin_path(run_dir: Path) -> Path:
    """The C++-loadable native .bin twin of the servable net (best.bin / weights.bin) —
    what the no-Python self-play client fetches by sha. Published by the trainer
    (train_continuous._publish) alongside the .pt; '' sha if it doesn't exist yet."""
    return _net_path(run_dir).with_suffix(".bin")


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


def _gate_stats(run_dir: Path) -> tuple[int, float]:
    """Promotions so far + the latest cumulative best Elo, read from the arena's
    gate_log.jsonl (shared run-dir). (0, 0.0) before the first gate. Cheap enough
    for the human-polled /status — the log is one line per gate cycle."""
    promotions = 0
    best_elo = 0.0
    try:
        for line in (run_dir / "gate_log.jsonl").read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("promoted"):
                promotions += 1
                if "best_elo" in rec:
                    best_elo = float(rec["best_elo"])
    except OSError:
        pass
    return promotions, best_elo


# --- content-addressed nets + lc0 multipart upload (Phase A: lc0-canonical wire) -------
_SHA_CACHE: dict = {}          # path -> ((mtime, size), sha256hex) — avoid rehashing the net every poll
_SHA_LOCK = threading.Lock()


def _file_sha(path: Path) -> str | None:
    """sha256 hex of a file, cached by (mtime, size) so repeat polls don't rehash the net.
    None if the file is absent/unreadable. This is the content address lc0 keys networks by."""
    try:
        st = path.stat()
    except OSError:
        return None
    key = (st.st_mtime, st.st_size)
    with _SHA_LOCK:
        c = _SHA_CACHE.get(path)
        if c is not None and c[0] == key:
            return c[1]
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    sha = h.hexdigest()
    with _SHA_LOCK:
        _SHA_CACHE[path] = (key, sha)
    return sha


def _net_files(run_dir: Path) -> list:
    """Every net the server can serve by sha: the gated champion, the raw weights, the
    candidate, and each archived gate opponent. Order is the preferred-match order."""
    files = []
    for name in ("best.pt", "weights.pt", "cand.pt",
                 "best.bin", "weights.bin", "cand.bin"):  # .bin = C++-client nets (Phase 3B)
        p = run_dir / name
        if p.exists():
            files.append(p)
    mn = run_dir / "match_nets"
    if mn.exists():
        files.extend(sorted(mn.glob("*.pt")))
        files.extend(sorted(mn.glob("*.bin")))
    return files


def _resolve_sha(run_dir: Path, sha: str) -> Path | None:
    """The servable net file whose content sha256 == `sha`, else None — lc0's get_network:
    a client asks for a net by hash, immune to which filename it currently lives under."""
    for p in _net_files(run_dir):
        if _file_sha(p) == sha:
            return p
    return None


def _selfplay_params(run_dir: Path) -> dict:
    """run_dir/selfplay.json as a dict ({} if missing/invalid). Embedded in the `train` job
    so a client gets net sha + params in a single next_game round-trip (lc0-style)."""
    try:
        return json.loads((run_dir / "selfplay.json").read_text())
    except (OSError, ValueError):
        return {}


def _parse_multipart(content_type: str, body: bytes) -> dict:
    """Minimal multipart/form-data parser (stdlib only — py3.13 removed `cgi`). Returns
    {field_name: (filename_or_None, raw_bytes)}. Sufficient for the fixed parts the fleet
    client posts (filename / trainingdata / meta); not a general RFC 7578 implementation."""
    if "multipart/form-data" not in content_type:
        raise ValueError("not multipart/form-data")
    m = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type)
    if not m:
        raise ValueError("no boundary")
    boundary = (m.group(1) or m.group(2)).strip().encode()
    out: dict = {}
    for seg in body.split(b"--" + boundary):
        # Each part is framed by a CRLF after the delimiter line and a CRLF before the next
        # delimiter. Trim EXACTLY those two (never strip(), which would eat a binary payload's
        # own trailing \r/\n). Skip the preamble / closing '--' terminator.
        if not seg or seg in (b"--", b"--\r\n", b"\r\n"):
            continue
        if seg.startswith(b"\r\n"):
            seg = seg[2:]
        if seg.endswith(b"\r\n"):
            seg = seg[:-2]
        head, sep, content = seg.partition(b"\r\n\r\n")
        if not sep:
            continue
        name = filename = None
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-disposition"):
                hn = re.search(rb'name="([^"]*)"', line)
                hf = re.search(rb'filename="([^"]*)"', line)
                name = hn.group(1).decode("utf-8", "replace") if hn else None
                filename = hf.group(1).decode("utf-8", "replace") if hf else None
        if name is not None:
            out[name] = (filename, content)
    return out


class _Handler(BaseHTTPRequestHandler):
    server_version = "ChesskersFleet/1.0"
    run_dir: Path  # set on the server instance below; bound per-request via self.server

    def _run_dir(self) -> Path:
        return self.server.run_dir  # type: ignore[attr-defined]

    def _note_client(self) -> None:
        """Stamp the calling box's last-seen time, code version (X-Client-Version), and
        worker-subprocess liveness (X-Client-Workers: up/down/off, when the client owns
        its workers) for the /status fleet view. Best-effort and header-only — a client
        that sends no X-Client-Id is invisible here. A request lacking a given header
        keeps the last value seen for that box, so a plain GET doesn't clobber state."""
        cid = self.headers.get("X-Client-Id")
        if not cid:
            return
        cid = cid[:64]
        ver = self.headers.get("X-Client-Version")
        wk = self.headers.get("X-Client-Workers")
        net = self.headers.get("X-Client-Net")  # content sha of the net this box reports running
        with self.server.clients_lock:  # type: ignore[attr-defined]
            prev = self.server.clients.get(cid)  # type: ignore[attr-defined]
            self.server.clients[cid] = (  # type: ignore[attr-defined]
                time.time(),
                ver or (prev[1] if prev else None),
                wk or (prev[2] if prev else None),
                net or (prev[3] if prev else None),
            )

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

    def _read_body(self) -> bytes:
        """The request body, bounded by Content-Length (<= MAX_UPLOAD_BYTES). b'' if absent,
        non-positive, or over the cap — callers treat b'' as 'nothing to ingest'."""
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return b""
        if n <= 0 or n > MAX_UPLOAD_BYTES:
            return b""
        return self.rfile.read(n)

    def _atomic_write(self, target: Path, data: bytes) -> bool:
        """Write `data` to `target` atomically (tmp `.part` + os.replace) so the trainer's
        drain never globs a half-written file. True on success; logs + False on OSError."""
        tmp = target.with_name(target.name + ".part")
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, target)
            return True
        except OSError as e:
            log.warning("write %s failed: %s", target.name, e)
            return False

    def do_GET(self) -> None:
        self._note_client()
        rd = self._run_dir()
        if self.path == "/version":
            self._send(200, _version(rd).encode())
        elif self.path == "/control":
            # X-Network-Sha rides the heartbeat tick the client already makes: the current
            # net's content address, so the client fetches it via /get_network without an
            # extra round-trip (lc0 content-addressed sync). '' before the first net exists.
            body = b"STOP" if (rd / "STOP").exists() else b"RUN"
            # X-Network-Bin-Sha = the C++-loadable .bin twin's content address (Phase 3B-3:
            # the orchestrator syncs run-dir/weights.bin off this for the cc_selfplay engine,
            # exactly as it syncs weights.pt off X-Network-Sha). '' until a .bin is published;
            # additive, so Python clients that only read X-Network-Sha are unaffected.
            self._send(200, body, extra={"X-Network-Sha": _file_sha(_net_path(rd)) or "",
                                         "X-Network-Bin-Sha": _file_sha(_net_bin_path(rd)) or ""})
        elif self.path == "/client-version":
            self._send(200, self.server.code_version.encode())  # type: ignore[attr-defined]
        elif self.path == "/selfplay":
            try:
                data = (rd / "selfplay.json").read_bytes()
            except OSError:
                data = b"{}"  # nothing published -> clients keep their launch defaults
            self._send(200, data, "application/json")
        elif self.path == "/weights":
            net = _net_path(rd)
            try:
                data = net.read_bytes()
            except OSError:
                self._send(404, b"no weights yet")
                return
            self._send(200, data, "application/octet-stream",
                       {"X-Version": _version(rd), "X-Network-Sha": _file_sha(net) or ""})
        elif self.path == "/status":
            backlog = sum(1 for _ in (rd / "buffer").glob("*.pkl")) if (rd / "buffer").exists() else 0
            m = _read_match(rd)
            promotions, best_elo = _gate_stats(rd)
            now = time.time()
            with self.server.clients_lock:  # type: ignore[attr-defined]
                clients = {cid: {"age": round(now - ts, 1), "version": ver, "workers": wk, "net": net}
                           for cid, (ts, ver, wk, net) in self.server.clients.items()  # type: ignore[attr-defined]
                           if now - ts <= CLIENT_ACTIVE_WINDOW}
            with self.server.stats_lock:  # type: ignore[attr-defined]
                games_ingested = self.server.games_ingested  # type: ignore[attr-defined]
            body = json.dumps({
                "version": _version(rd),
                "net_sha": _file_sha(_net_path(rd)) or "",  # content address of the served net (lc0 get_network)
                "weights": (rd / "weights.pt").exists(),
                "best": (rd / "best.pt").exists(),
                "buffer_backlog": backlog,
                "control": "STOP" if (rd / "STOP").exists() else "RUN",
                "gate_open": bool(m),
                "match_id": (m or {}).get("match_id"),
                "games_ingested": games_ingested,  # game pkls POSTed to this process since start
                "promotions": promotions,
                "best_elo": round(best_elo, 1),
                "clients_active": len(clients),
                "clients": clients,  # {id: {age: s-since-seen, version: git-sha, workers: up/down/off, net: sha}}, active window only
            }).encode()
            self._send(200, body, "application/json")
        elif self.path.split("?", 1)[0] == "/get_network":
            # lc0-canonical content-addressed fetch: GET /get_network?sha=<sha256>. Serves
            # whichever servable net hashes to <sha> (best/weights/cand/opponents); 404 on miss.
            sha = parse_qs(urlsplit(self.path).query).get("sha", [""])[0]
            p = _resolve_sha(rd, sha) if sha else None
            if p is None:
                self._send(404, b"no such network")
                return
            try:
                data = p.read_bytes()
            except OSError:
                self._send(404, b"no such network")
                return
            self._send(200, data, "application/octet-stream", {"X-Network-Sha": sha})
        else:
            self._send(404, b"not found")

    def do_POST(self) -> None:
        self._note_client()
        if self.path == "/match_result":
            self._match_result()
            return
        if self.path == "/next_game":      # lc0-canonical job assignment (train vs match)
            self._next_game_post()
            return
        if self.path == "/upload_game":    # lc0-canonical multipart upload (alongside POST /game)
            self._upload_game()
            return
        if not self.path.startswith("/game/"):
            self._send(404, b"not found")
            return
        name = self.path[len("/game/"):]
        if not _NAME_RE.match(name):
            self._send(400, b"bad filename")
            return
        data = self._read_body()
        if not data:
            self._send(400, b"bad length")
            return
        buf = self._run_dir() / "buffer"
        buf.mkdir(parents=True, exist_ok=True)
        # Atomic for the .pkl so the trainer's drain never sees a half-written game
        # (it globs *.pkl and decodes each ccz chunk; see training_chunk). .meta: small write.
        if not self._atomic_write(buf / name, data):
            self._send(500, b"write failed")
            return
        if name.endswith(".pkl"):  # count games (not the .meta sidecar) for /status throughput
            with self.server.stats_lock:  # type: ignore[attr-defined]
                self.server.games_ingested += 1  # type: ignore[attr-defined]
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
                           "opp": str(r.get("opp") or "best")[:64], "outcome": r["outcome"],
                           "match_id": m["match_id"]}).encode()
        if not self._atomic_write(target, body):
            self._send(500, b"write failed")
            return
        self._send(200, b"ok")

    def _next_game_post(self) -> None:
        """lc0-canonical job assignment (POST /next_game) — the sole assignment path. Selects
        from match.json + round-robin over the gate panel (via the shared match_cursor) and
        replies in lc0's vocabulary: {"type":"train"|"match","sha":<net sha>,"params":...}.
        Nets are identified by sha256 so the client fetches them with GET /get_network?sha=.
        Any POSTed body (lc0 clients send their identity) is drained and ignored — liveness
        still rides the X-Client-Id header."""
        rd = self._run_dir()
        self._read_body()  # drain/ignore posted form fields
        m = _read_match(rd)
        seeds = (m or {}).get("seeds") or []
        opps = (m or {}).get("opponents") or ["best"]
        if not m or not seeds:
            # `sha` = the .pt net (Python clients); `bin_sha` = its C++-loadable .bin
            # twin (the no-Python client fetches THIS by sha). bin_sha is "" until the
            # trainer has published a .bin — additive, never breaks the .pt path.
            job = {"type": "train", "sha": _file_sha(_net_path(rd)) or "",
                   "bin_sha": _file_sha(_net_bin_path(rd)) or "",
                   "params": _selfplay_params(rd)}
            self._send(200, json.dumps(job).encode(), "application/json")
            return
        units = [(o, s, cw) for o in opps for s in seeds for cw in (True, False)]
        opp, seed, cand_white = units[next(self.server.match_cursor) % len(units)]  # type: ignore[attr-defined]
        cand = rd / "cand.pt"
        opp_path = (rd / "best.pt") if opp == "best" else (rd / "match_nets" / f"{opp}.pt")
        job = {
            "type": "match", "match_id": m["match_id"],
            "sha": _file_sha(cand) or "", "candidate_sha": _file_sha(cand) or "",
            "opponent": opp, "opponent_sha": _file_sha(opp_path) or "",
            # .bin twins (Phase 4): the C++ gate client fetches both nets by these
            # (additive — Python clients use candidate_sha/opponent_sha; "" if absent).
            "candidate_bin_sha": _file_sha(cand.with_suffix(".bin")) or "",
            "opponent_bin_sha": _file_sha(opp_path.with_suffix(".bin")) or "",
            "seed": seed, "cand_white": cand_white,
            "arch": m["arch"], "params": m["params"],
        }
        self._send(200, json.dumps(job).encode(), "application/json")

    def _upload_game(self) -> None:
        """lc0-canonical multipart game upload (POST /upload_game). Parses multipart/form-data
        (stdlib; py3.13 dropped cgi) and lands the game into buffer/ exactly like POST /game,
        so the trainer's drain is unchanged. Parts: filename (NNN_..pkl, validated — no path
        traversal), trainingdata (the game bytes — a gzipped-JSON `ccz` chunk since Phase C),
        meta (optional .pkl.meta). This endpoint is a byte pipe: it never parses the payload,
        so the record schema lives entirely in the worker (write) + trainer (drain; see
        training_chunk)."""
        rd = self._run_dir()
        body = self._read_body()
        if not body:
            self._send(400, b"empty body")
            return
        try:
            parts = _parse_multipart(self.headers.get("Content-Type", ""), body)
        except ValueError:
            self._send(400, b"bad multipart")
            return
        fn = parts.get("filename")
        filename = fn[1].decode("utf-8", "replace").strip() if fn else ""
        if not _NAME_RE.match(filename) or not filename.endswith(".pkl"):
            self._send(400, b"bad filename")
            return
        td = parts.get("trainingdata") or parts.get("file")
        if not td or not td[1]:
            self._send(400, b"no trainingdata")
            return
        buf = rd / "buffer"
        buf.mkdir(parents=True, exist_ok=True)
        meta = parts.get("meta")
        if meta and meta[1]:  # meta first, like /game, so the drain never sees a meta-less pkl
            self._atomic_write(buf / (filename + ".meta"), meta[1])
        if not self._atomic_write(buf / filename, td[1]):
            self._send(500, b"write failed")
            return
        with self.server.stats_lock:  # type: ignore[attr-defined]
            self.server.games_ingested += 1  # type: ignore[attr-defined]
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
    httpd.clients = {}                        # X-Client-Id -> (last-seen epoch, version, workers, net sha) (fleet liveness)
    httpd.clients_lock = threading.Lock()     # guards clients across handler threads
    httpd.games_ingested = 0                  # cumulative game pkls ingested this process (/status throughput)
    httpd.stats_lock = threading.Lock()       # guards games_ingested across handler threads
    httpd.code_version = _code_version()      # this host's git sha, served at /client-version for self-update
    httpd.daemon_threads = True
    log.info("fleet server up on %s:%d (run-dir=%s, version=%s, code=%s)",
             args.host, args.port, run_dir, _version(run_dir), httpd.code_version)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("fleet server stopping")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
