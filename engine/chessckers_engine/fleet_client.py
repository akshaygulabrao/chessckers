"""Fleet client — lc0-style orchestrator (owns the cc_selfplay engine pool).

Runs on every self-play box (the local machine over loopback AND leena / future
volunteers over the LAN). This is the lc0 ORCHESTRATOR (the lczero-client analog): the
ONLY thing that talks to the server, owning a pool of native `cc_selfplay --jobs-local`
ENGINE procs (the lc0 engine-binary analog). The *client-drives-each-game* model — the
server assigns EVERY game: the client keeps the engines' run-dir job queue topped up by
POSTing /next_game, and the engines claim + play one assigned game at a time (train -> a
self-play game into buffer/; match -> a keep-best gate game into match_out/). There is NO
autonomous self-play — a box plays exactly what /next_game hands out, so the trainer
controls the train/gate mix fleet-wide. A volunteer needs no inbound SSH (just outbound
HTTP) and the engines themselves never touch the network. It replaces the rsync
`leena_sync.sh` sidecar.

It spawns the pool once the net lands, restarts dead engines, and heartbeats its
liveness — so a zombie box (client up, engines dead, producing nothing) shows up in
/status instead of silently lying.

Each tick (`--poll-seconds`):
  1. GET /control (carrying `X-Client-Id` + `X-Client-Version` heartbeat headers so
     the server can list live boxes AND their code version in /status — a box on a
     stale commit is then visible, not a silent runtime crash) — if STOP, signal the
     local engines (touch run-dir/STOP) and exit. A tick that raises an unexpected
     error is logged and retried next poll — the loop never dies (an unattended
     volunteer box must outlive transient faults).
  1b. GET /client-version — the trainer host's code sha. With --update-cmd set, a
     mismatch against the sha this client booted on triggers a pull + native rebuild
     and an os.execv re-exec onto the fresh code (retried with backoff so it self-
     corrects even if the code lands here just after the server advertises it). Without
     --update-cmd the drift is only logged, so a stale box stays visible in /status.
  2. Net sync (content-addressed, lc0 get_network): /control's response carries an
     X-Network-Sha header (the .pt net's sha256, for the fleet net-consistency view) and
     an X-Network-Bin-Sha header (the C++-loadable .bin twin's sha256); on a change, GET
     /get_network?sha= and write run-dir/weights.{pt,bin} atomically. The cc_selfplay
     engines hot-reload weights.bin on their own mtime poll. The client gets back exactly
     the bytes it asked for, keyed by hash. Falls back to the legacy /version+/weights
     poll against a server that doesn't advertise the sha. Fires once per trainer
     ITERATION / promotion.
  3. Upload finished self-play games (train job output): each settled buffer/*.pkl (a
     gzipped-JSON `ccz` chunk, not a pickle — moved opaquely) is POSTed with its `.meta`
     in ONE multipart /upload_game request (server lands meta-before-pkl), then deleted —
     each game uploaded exactly once.
  3b. Ship finished gate outcomes (match job output): each match_out/*.json -> POST
     /match_result, then deleted.
  4. (--spawn-engines) Supervise the engine pool: spawn N `cc_selfplay --jobs-local` procs
     once weights.bin has landed, restart any that die, and heartbeat their state
     (X-Client-Workers: up/down/off) so /status can flag a zombie box. The run-dir job queue
     is reset once on first spawn so a fresh pool never claims a stale job (one minted against
     an old net / left by a crashed pool).
  5. Mint jobs (the lc0 next_game loop): while fewer than --queue-depth jobs sit unclaimed in
     run-dir/jobs/, POST /next_game and queue the reply for an engine to claim — a `train` job
     verbatim; a `match` job with the candidate + opponent nets pre-fetched by content address
     (/get_network?sha=) and their local .bin paths added, so the engine plays it without
     touching the network. Roughly one /next_game per game, the server arbitrating train vs
     gate every time. A self-play-only box (no gate deps) declines match jobs; the arena +
     engine boxes carry the gate.

The client is stdlib only (urllib) — no requests/aiohttp dep, so it runs on a bare
volunteer venv. It shells out to the cc_selfplay engine (never imports it), so the client
never pulls in torch / the native ext at import; the engine carries the rules+NN and plays
every game.

Run (on a self-play box):

    python -m chessckers_engine.fleet_client \\
      --server http://192.168.1.50:8000 --run-dir ~/chessckers/run --poll-seconds 15 \\
      --queue-depth 4 --spawn-engines --engine-workers 4 \\
      --engine-binary engine/cpp/build/cc_selfplay --engine-worker-id-base 300
"""
from __future__ import annotations

import argparse
import errno
import http.client
import json
import logging
import os
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from chessckers_engine.runtime import setup_logging

log = logging.getLogger("chessckers_engine.fleet_client")

UPDATE_RETRY_S = 300.0  # re-attempt a failed/too-early self-update to a given sha no more often than this
HEARTBEAT_S = 600.0     # while healthy, log a "still alive" health line no more often than this (transitions log immediately)
DEEP_DIAG_S = 60.0      # while unreachable, emit the kitchen-sink _deep_diag at most this often (onset always dumps)

_IP_BOUND_IF = 25  # macOS <netinet/in.h>: pins a socket to an interface, overriding VPN/tunnel scoping
_opener: urllib.request.OpenerDirector | None = None  # set by --bind-interface; None = default routing


def _build_bound_opener(ifname: str) -> urllib.request.OpenerDirector:
    """An opener whose HTTP connections are pinned to `ifname` via macOS IP_BOUND_IF. On a box
    where a VPN / Private Relay scopes the process's outbound TCP onto a `utun` (which can't reach
    the LAN server -> 'No route to host' / 'Network is unreachable'), this forces the socket onto
    the real LAN interface so the server stays reachable. macOS-only; opt-in per box."""
    idx = socket.if_nametoindex(ifname)

    class _BoundConn(http.client.HTTPConnection):
        def connect(self):  # type: ignore[override]
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_IP, _IP_BOUND_IF, struct.pack("I", idx))
            if self.timeout is not None and self.timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                s.settimeout(self.timeout)
            s.connect((self.host, self.port))
            self.sock = s

    class _BoundHandler(urllib.request.HTTPHandler):
        def http_open(self, req):
            return self.do_open(_BoundConn, req)

    return urllib.request.build_opener(_BoundHandler), idx  # idx = the cached iface index, for drift checks


def _urlopen(req: urllib.request.Request, timeout: float):
    """urlopen via the interface-bound opener when --bind-interface is set, else the stdlib default."""
    if _opener is not None:
        return _opener.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def _git_version() -> str:
    """Best-effort short git sha of the running client code (resolved once at
    startup), reported so a box on a STALE commit — the silent-runtime-crash class
    we hit with the mismatched native .so — shows up plainly in the server's
    /status. 'unknown' when not run from a git checkout."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 — version is diagnostics-only; never block the client on it
        return "unknown"


def _head_contains(sha: str) -> bool:
    """True if the running tree already CONTAINS `sha` (it's an ancestor of HEAD) — i.e.
    this box is AT or AHEAD of that commit. Used to suppress a self-update that would
    DOWNGRADE the box: a freshly-deployed client (new sha) polling a server still on an
    older boot sha sees the drift BACKWARDS and must ignore it (a downgrade would also
    needlessly thrash the owned workers). Conservative — any git error returns False, so a
    genuinely-behind box still updates forward."""
    try:
        r = subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"],
                           capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:  # noqa: BLE001 — best-effort; fall through to the normal update path
        return False


def _net_diag(server: str, bind_iface: str) -> str:
    """Best-effort one-line snapshot of THIS box's egress state, captured at the moment a
    request to `server` FAILS — the single most useful signal for the intermittent
    LAN-unreachable / VPN-socket-scoping class we're chasing. Reports the default route the
    kernel would now pick to the server host, whether the pinned interface still owns an
    IPv4, and any tunnels up. macOS tools (route/ifconfig); never raises — returns '' when it
    can't be gathered (non-mac / tool missing) so it can be dropped straight into a log line."""
    try:
        host = urlparse(server).hostname or server
    except Exception:  # noqa: BLE001 — diagnostics only; never let this perturb the tick
        host = server
    parts = []
    try:
        out = subprocess.run(["route", "-n", "get", host],
                             capture_output=True, text=True, timeout=3).stdout
        dev = next((l.split(":", 1)[1].strip() for l in out.splitlines()
                    if l.strip().startswith("interface:")), "?")
        parts.append(f"route->{host}=dev:{dev}")
    except Exception:  # noqa: BLE001
        pass
    if bind_iface:
        try:
            out = subprocess.run(["ifconfig", bind_iface],
                                 capture_output=True, text=True, timeout=3).stdout
            ip = next((l.split()[1] for l in out.splitlines()
                       if l.strip().startswith("inet ") and "inet6" not in l), None)
            parts.append(f"{bind_iface}={ip or 'NO-IP'}")
        except Exception:  # noqa: BLE001
            pass
    try:
        out = subprocess.run(["ifconfig", "-l"], capture_output=True, text=True, timeout=3).stdout
        tuns = [n for n in out.split() if n.startswith(("utun", "ipsec", "ppp"))]
        parts.append("tun:" + (",".join(tuns) if tuns else "none"))
    except Exception:  # noqa: BLE001
        pass
    return " ".join(parts)


def _err_detail(e: BaseException) -> str:
    """'<ExcType>/<ERRNONAME>(<n>): <msg>' — unwraps urllib.error.URLError to the underlying
    OSError so the symbolic errno (EHOSTUNREACH, ENETUNREACH, ETIMEDOUT, ECONNREFUSED …) is
    explicit in the log rather than a bare number you have to look up."""
    inner = getattr(e, "reason", None) or e
    en = getattr(inner, "errno", None)
    name = errno.errorcode.get(en, "?") if isinstance(en, int) else "?"
    return f"{type(e).__name__}/{name}({en}): {inner}"


def _raw_connect(host: str, port: int, ifname: str | None, timeout: float = 2.0) -> str:
    """Fresh in-process TCP connect attempt (optionally IP_BOUND_IF-pinned) so a failure dump
    records whether a brand-new socket reaches the server AT THIS INSTANT and which local
    address the kernel picks. Never raises."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        if ifname:
            s.setsockopt(socket.IPPROTO_IP, _IP_BOUND_IF, struct.pack("I", socket.if_nametoindex(ifname)))
        t0 = time.time()
        s.connect((host, port))
        return "OK src=%s:%d %.0fms" % (*s.getsockname(), (time.time() - t0) * 1000)
    except OSError as e:
        return f"FAIL {errno.errorcode.get(e.errno, '?')}({e.errno}) {e.strerror}"
    finally:
        s.close()


def _raw_connect_idx(host: str, port: int, idx: int, timeout: float = 2.0) -> str:
    """Like _raw_connect but pinned to an explicit interface INDEX (not re-resolved from a name) —
    i.e. the EXACT value the bound opener cached at startup. If this FAILS while the fresh-name
    pin SUCCEEDS, the cached index is stale and that's the bug (the opener holds the only copy)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.setsockopt(socket.IPPROTO_IP, _IP_BOUND_IF, struct.pack("I", idx))
        t0 = time.time()
        s.connect((host, port))
        return "OK src=%s:%d %.0fms" % (*s.getsockname(), (time.time() - t0) * 1000)
    except OSError as e:
        return f"FAIL {errno.errorcode.get(e.errno, '?')}({e.errno}) {e.strerror}"
    finally:
        s.close()


def _sh(cmd: list, timeout: float = 3.0) -> str:
    """Best-effort capture of a short shell command's stdout; '(err …)' instead of raising."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:  # noqa: BLE001 — diagnostics only
        return f"(err {e})"


def _deep_diag(server: str, bind_iface: str, cached_idx) -> str:
    """Kitchen-sink egress snapshot for a 'No route to host' (EHOSTUNREACH) that only bites the
    long-running client. Best-effort, never raises; macOS tools, no sudo. Captures the things
    that actually explain an L2/route-unreachable: the ARP/neighbor entry for the server (the
    crux), the full host route (flags/expire), the bound interface's link state, AWDL radio
    status, the TCP socket-state census (TIME_WAIT / ephemeral-port pile-up), memory+swap
    pressure (the 8GB Air runs 4 torch workers), the live-vs-cached iface index, and fresh
    pinned+unpinned connects with the errno + local address the kernel picks right now."""
    u = urlparse(server)
    host = u.hostname or server
    port = u.port or 80
    iface = bind_iface or "en0"
    L = []
    # ARP / neighbor — the crux for EHOSTUNREACH on a directly-connected LAN
    L.append("arp:     " + (_sh(["arp", "-n", host]) or "(no entry)"))
    # full host route — keep the informative fields
    rt = _sh(["route", "-n", "get", host])
    L.append("route:   " + " ".join(s.strip() for s in rt.splitlines()
             if any(k in s for k in ("interface:", "gateway:", "flags:", "expire:"))))
    # bound interface link state — UP/RUNNING/ACTIVE? still owns its inet? status active/inactive?
    eg = _sh(["ifconfig", iface])
    head = eg.splitlines()[0] if eg else ""
    inet = next((s.strip() for s in eg.splitlines() if s.strip().startswith("inet ")), "(no inet)")
    status = next((s.strip() for s in eg.splitlines() if "status:" in s), "")
    L.append(f"{iface}:    {head.split('mtu')[0].strip()} | {inet} | {status}")
    # AWDL radio — AirDrop/Handoff/Continuity time-shares the Wi-Fi channel
    aw = _sh(["ifconfig", "awdl0"])
    awf = aw.split("flags=")[1].split("<")[0] if "flags=" in aw else "?"
    aws = next((s.strip() for s in aw.splitlines() if "status:" in s), "")
    L.append(f"awdl0:   flags={awf} {aws}")
    # TCP socket census + conns to the server (TIME_WAIT churn / ephemeral-port exhaustion)
    ns = _sh(["netstat", "-an", "-p", "tcp"])
    states: dict = {}
    nsrv = 0
    for r in (s.split() for s in ns.splitlines()):
        if len(r) >= 6 and r[-1].isupper():
            states[r[-1]] = states.get(r[-1], 0) + 1
        if any(f"{host}.{port}" in tok for tok in r):
            nsrv += 1
    L.append(f"tcp:     {states}  conns->{host}.{port}={nsrv}")
    # memory + swap pressure (4 torch workers on an 8GB box — level 1=normal 2=warn 4=critical)
    L.append("mem:     pressure_lvl=" + _sh(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]) +
             "  " + _sh(["sysctl", "-n", "vm.swapusage"]))
    # iface index now vs the value the bound opener cached at startup
    try:
        live = socket.if_nametoindex(iface)
        L.append(f"idx:     {iface} live={live} cached={cached_idx}" +
                 ("  <-- DRIFTED" if cached_idx is not None and live != cached_idx else ""))
    except OSError as e:
        L.append(f"idx:     err {e}")
    # fresh connects RIGHT NOW — the discriminator. unpinned; pinned by NAME (re-resolves the
    # index live); and pinned to the EXACT index the opener cached at startup. cached-FAIL +
    # fresh-OK == stale cached index (the opener is the only holder); both-OK == not the socket
    # layer at all (the urllib opener state is suspect instead).
    L.append(f"connect unpin:           {_raw_connect(host, port, None)}")
    L.append(f"connect pin {iface}(fresh):  {_raw_connect(host, port, iface)}")
    if cached_idx is not None:
        L.append(f"connect pin idx={cached_idx}(cached): {_raw_connect_idx(host, port, cached_idx)}")
    return "\n    ".join(L)


class _EnginePool:
    """Supervise N `cc_selfplay --jobs-local` engine procs — the lc0 engine-binary half the
    orchestrator drives (Phase 3B-3 cutover). Each proc claims jobs from run_dir/jobs/ (the N
    procs race-claim the shared queue via atomic rename), plays the native engine off
    run_dir/weights.bin, and writes buffer/ + match_out/. Each gets a distinct --worker-id and
    --seed (so chunk filenames + RNG streams don't collide). Dead procs are restarted
    individually (a crashed engine only frees its one in-flight claim). The lc0-style
    orchestrator (HTTP / STOP / self-update / heartbeat) stays in fleet_client; this is just the
    engine the client owns, swapped Python->C++."""

    def __init__(self, binary: str, run_dir: Path, n: int, worker_id_base: int, seed: int,
                 machine: str, log_dir: Path, batch_size: int = 1, use_gpu: bool = False) -> None:
        self.binary = binary
        self.run_dir = run_dir
        self.n = max(1, n)
        self.worker_id_base = worker_id_base
        self.seed = seed
        self.machine = machine
        self.log_dir = log_dir
        self.batch_size = max(1, batch_size)  # >1 => each engine GPU/CPU-batches its self-play
        self.use_gpu = use_gpu
        self.procs: list = [None] * self.n
        self.logs: list = [None] * self.n

    def _spawn_one(self, i: int) -> None:
        wid = self.worker_id_base + i
        cmd = [self.binary, "--jobs-local", "--run-dir", str(self.run_dir),
               "--worker-id", str(wid), "--seed", str(self.seed + i), "--machine", self.machine]
        if self.batch_size > 1:
            cmd += ["--batch-size", str(self.batch_size)]
        if self.use_gpu:
            cmd.append("--gpu")
        f = open(self.log_dir / f"engine-{wid}.log", "a")
        # cc_selfplay reads CHESSCKERS_START_FEN from the inherited env (the launcher exports it).
        self.procs[i] = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        self.logs[i] = f
        log.info("spawned engine %d (pid %d, cc_selfplay --jobs-local)", wid, self.procs[i].pid)

    def ensure(self) -> None:
        """Spawn any engine that isn't running (first start) or has died (restart)."""
        for i in range(self.n):
            p = self.procs[i]
            if p is None:
                self._spawn_one(i)
            elif p.poll() is not None:
                log.warning("engine %d exited (rc=%s) — restarting", self.worker_id_base + i,
                            p.returncode)
                if self.logs[i] is not None:
                    self.logs[i].close()
                self._spawn_one(i)

    def started(self) -> bool:
        return any(p is not None for p in self.procs)

    def any_alive(self) -> bool:
        return any(p is not None and p.poll() is None for p in self.procs)

    def status(self) -> str:
        """Heartbeat tag (X-Client-Workers): off (none spawned) / up (all alive) / down (any dead)."""
        if not self.started():
            return "off"
        return "up" if all(p is not None and p.poll() is None for p in self.procs) else "down"

    def stop(self, timeout: float = 30.0) -> None:
        """Wind the pool down: the engines self-exit on run_dir/STOP (touched by the caller);
        wait, then terminate any straggler."""
        for p in self.procs:
            if p is None:
                continue
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.terminate()
            except OSError:
                pass


def _self_update(want: str, current: str, update_cmd: str, run_dir: Path,
                 stop_path: Path, engine_pool=None) -> None:
    """Pull + rebuild this box onto code sha `want` via `update_cmd`, then re-exec this
    client onto the fresh code (lc0's client-updates-itself, adapted). On success
    os.execv replaces this process and never returns. On failure (cmd errored, or the
    working tree didn't actually advance to `want`) it logs and returns, leaving the box
    visibly on `current` — the caller's backoff retries later (e.g. once the code lands).
    The owned engine pool is wound down before the rebuild so a stale binary isn't
    mid-search."""
    log.warning("code self-update: server on %s, running %s — running update-cmd", want, current)
    if engine_pool is not None:
        try:
            stop_path.touch()
        except OSError:
            pass
        engine_pool.stop(timeout=5.0)
    try:
        r = subprocess.run(update_cmd, shell=True, cwd=str(run_dir.parent),
                           timeout=600, capture_output=True, text=True)
    except (subprocess.TimeoutExpired, OSError) as e:
        log.error("update-cmd failed to run (%s) — staying on %s", e, current)
        return
    try:
        (run_dir / "update.log").write_text(
            f"$ {update_cmd}\nrc={r.returncode}\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    except OSError:
        pass
    now_sha = _git_version()
    if r.returncode == 0 and now_sha == want:
        log.warning("update OK; box now on %s — re-exec client onto fresh code", now_sha)
        try:
            stop_path.unlink()  # clear the wind-down sentinel for the fresh workers
        except OSError:
            pass
        os.execv(sys.executable,
                 [sys.executable, "-m", "chessckers_engine.fleet_client", *sys.argv[1:]])
    log.error("update-cmd rc=%d, tree now %s (wanted %s) — staying on %s; see run/update.log",
              r.returncode, now_sha, want, current)


def _get(url: str, timeout: float, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    with _urlopen(req, timeout) as r:
        return r.read()


def _post(url: str, data: bytes, timeout: float,
          content_type: str = "application/octet-stream") -> None:
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": content_type})
    with _urlopen(req, timeout) as r:
        r.read()


def _post_recv(url: str, data: bytes, timeout: float, headers: dict | None = None,
               content_type: str = "application/json") -> bytes:
    """POST and RETURN the response body — the lc0 `next_game` (job JSON reply) and
    `match_result` calls both need the reply and carry the X-Client-* heartbeat headers.
    Goes through the interface-bound opener like every other request."""
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": content_type, **(headers or {})})
    with _urlopen(req, timeout) as r:
        return r.read()


def _build_multipart(parts: list[tuple[str, str | None, bytes]]) -> tuple[str, bytes]:
    """Build a multipart/form-data (content_type, body) from (name, filename, bytes)
    parts. Mirrors exactly what fleet_server._parse_multipart expects — stdlib only
    (no `requests`), so the client stays torch/dep-free. CRLF framing; the binary
    payload is embedded verbatim (no transfer-encoding). The boundary is a fixed
    ASCII token — a gzip game chunk won't contain it."""
    boundary = "----ccFleetUpload7Q2x"
    out = bytearray()
    for name, filename, data in parts:
        disp = f'Content-Disposition: form-data; name="{name}"'
        if filename is not None:
            disp += f'; filename="{filename}"'
        out += f"--{boundary}\r\n".encode() + disp.encode() + b"\r\n\r\n" + data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return f"multipart/form-data; boundary={boundary}", bytes(out)


def _get2(url: str, timeout: float, headers: dict | None = None):
    """Like _get but also returns the response headers (the case-insensitive
    http.client.HTTPMessage) — used to read X-Network-Sha off /control so the net's
    content address rides the heartbeat tick the client already makes."""
    req = urllib.request.Request(url, headers=headers or {})
    with _urlopen(req, timeout) as r:
        return r.read(), r.headers


def _pull_weights_if_new(server: str, weights: Path, last_version: str, timeout: float) -> str:
    """Pull weights.pt iff the server's version changed. Returns the (possibly
    unchanged) current version so the caller can track it."""
    try:
        version = _get(f"{server}/version", timeout).decode().strip()
    except (urllib.error.URLError, OSError) as e:
        log.warning("version poll failed (server down/restarting?): %r", e)
        return last_version
    if version == last_version or version == "none":
        return version
    try:
        data = _get(f"{server}/weights", timeout)
    except (urllib.error.URLError, OSError) as e:
        log.warning("weights fetch failed: %r", e)
        return last_version  # retry next tick
    tmp = weights.with_suffix(".pt.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, weights)  # atomic; bumps mtime -> local workers hot-reload
    except OSError as e:
        log.warning("weights write failed: %s", e)
        return last_version
    log.info("pulled net %s -> %s (%d KB)", version, weights, len(data) // 1024)
    return version


def _pull_sha_to(server: str, dest: Path, want_sha: str | None, have_sha: str,
                 timeout: float, label: str) -> str:
    """Content-addressed file sync (lc0 get_network): fetch the net whose sha256 is
    `want_sha` only when it differs from what we last wrote, and materialize it at `dest`
    (atomic; bumps mtime -> the engine hot-reloads). `want_sha` comes from a /control
    header, so an unchanged net costs nothing. Unlike the opaque /version, the client gets
    back EXACTLY the bytes it asked for (or a 404 if the net rotated mid-fetch -> retry
    next tick). Returns the sha now on disk (unchanged on empty want / no-op / fetch
    failure). Drives BOTH the .pt (Python workers) and the .bin twin (cc_selfplay)."""
    if not want_sha or want_sha == have_sha:
        return have_sha
    try:
        data = _get(f"{server}/get_network?sha={want_sha}", timeout)
    except (urllib.error.URLError, OSError) as e:
        log.warning("%s fetch (sha %s) failed: %r", label, want_sha[:12], e)
        return have_sha  # 404 (rotated) or server blip — retry next tick on the new sha
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, dest)  # atomic; bumps mtime -> engine hot-reloads
    except OSError as e:
        log.warning("%s write failed: %s", label, e)
        return have_sha
    log.info("pulled %s sha %s -> %s (%d KB)", label, want_sha[:12], dest, len(data) // 1024)
    return want_sha


def _pull_net_by_sha(server: str, weights: Path, want_sha: str | None, have_sha: str,
                     timeout: float) -> str:
    """The .pt net sync (Python workers hot-reload weights.pt) — thin wrapper over
    _pull_sha_to off the X-Network-Sha header."""
    return _pull_sha_to(server, weights, want_sha, have_sha, timeout, "net")


def _upload_games(server: str, buffer: Path, min_age: float, timeout: float) -> int:
    """POST each complete, settled game to the server via the lc0-canonical multipart
    /upload_game (parts: filename + trainingdata=the ccz chunk + optional meta), then
    delete it locally. The single request lands meta-before-pkl server-side, so the
    trainer never drains a pkl whose .meta is missing. Returns the count uploaded."""
    if not buffer.exists():
        return 0
    now = time.time()
    uploaded = 0
    for pkl in sorted(buffer.glob("*.pkl")):
        try:
            if now - pkl.stat().st_mtime < min_age:
                continue  # still being written / its .meta not yet flushed
        except OSError:
            continue
        meta = Path(str(pkl) + ".meta")
        try:
            parts: list[tuple[str, str | None, bytes]] = [
                ("filename", None, pkl.name.encode()),
                ("trainingdata", pkl.name, pkl.read_bytes()),
            ]
            if meta.exists():  # best-effort on the worker side; carry it when present
                parts.append(("meta", meta.name, meta.read_bytes()))
            ctype, body = _build_multipart(parts)
            _post(f"{server}/upload_game", body, timeout, ctype)
        except (urllib.error.URLError, OSError) as e:
            log.warning("upload %s failed (retry next tick): %r", pkl.name, e)
            break  # server down — stop this tick, keep games for retry
        for fp in (meta, pkl):
            try:
                fp.unlink()
            except OSError:
                pass
        uploaded += 1
    return uploaded


def _fetch_net(server: str, sha: str, cache_dir: Path, timeout: float,
               suffix: str = ".pt") -> Path | None:
    """Fetch a gate net by content address (lc0 get_network) into cache_dir/<sha><suffix>,
    skipping the download when it's already there (sha-named -> content-addressed, so a
    champion reused across gates isn't re-pulled). Routed through the interface-bound opener
    so leena's en0 pin is honored. `suffix` selects the twin: .pt for the Python MatchRunner,
    .bin for the cc_selfplay engine (same bytes either way — the server serves by sha). None
    on empty sha / fetch / write failure -> retry next tick (e.g. the net rotated mid-gate -> 404)."""
    if not sha:
        return None
    dest = cache_dir / f"{sha}{suffix}"
    if dest.exists():
        return dest
    try:
        data = _get(f"{server}/get_network?sha={sha}", timeout)
    except (urllib.error.URLError, OSError) as e:
        log.warning("gate net fetch (sha %s) failed: %r", sha[:12], e)
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, dest)
    except OSError as e:
        log.warning("gate net write failed: %s", e)
        return None
    return dest


def _reset_queue(jobs_dir: Path, match_out: Path) -> None:
    """Clear the run-dir job queue + any un-shipped gate outcomes — called once right before
    the engine pool's first spawn so it starts on an EMPTY queue: no stale claims left by a
    crashed pool, and no jobs minted against a now-superseded net. Best-effort per file."""
    for d in (jobs_dir, match_out):
        try:
            entries = list(d.glob("*"))
        except OSError:
            continue
        for f in entries:
            try:
                f.unlink()
            except OSError:
                pass


def _mint_jobs(server: str, jobs_dir: Path, gate_dir: Path, depth: int, start_seq: int,
               can_match: bool, timeout: float, headers: dict) -> int:
    """Top up the run-dir job queue so the owned workers always have server-assigned work — the
    lc0 client-drives-each-game loop. While fewer than `depth` jobs sit UNCLAIMED in jobs_dir,
    POST /next_game and write the reply as `<seq>.json` for a worker to claim (atomically, via a
    .tmp rename, so a worker never reads a half-file):
      • train -> {"type":"train","sha":…,"params":…} verbatim;
      • match -> the gate unit, with the candidate + opponent nets fetched by content address
                 (/get_network) into gate_dir and their local paths added (cand_path/opp_path),
                 so the worker plays it without ever touching the network.
    A self-play-only box (`can_match` False) declines a match job; a net not yet fetchable also
    stops this tick (retried next poll). Returns the next free sequence number."""
    try:
        jobs_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return start_seq
    seq = start_seq
    # Cap round-trips per tick so an all-match queue this box can't fill doesn't spin.
    for _ in range(max(1, depth) * 2):
        try:
            unclaimed = sum(1 for _ in jobs_dir.glob("*.json"))
        except OSError:
            break
        if unclaimed >= depth:
            break
        try:
            job = json.loads(_post_recv(f"{server}/next_game", b"", timeout, headers))
        except (urllib.error.URLError, OSError, ValueError) as e:
            log.debug("next_game poll failed: %r", e)
            break
        if job.get("type") == "match":
            if not can_match:
                break  # this box self-plays only; leave the gate to the arena / per-worker boxes
            cand = _fetch_net(server, job.get("candidate_sha", ""), gate_dir, timeout)
            opp = _fetch_net(server, job.get("opponent_sha", ""), gate_dir, timeout)
            if cand is None or opp is None:
                break  # net not ready/reachable — retry next tick
            job["cand_path"] = str(cand)
            job["opp_path"] = str(opp)
            # Additive: also fetch the C++-loadable .bin twins so a cc_selfplay --jobs-local
            # engine can play this gate game (it reads cand_bin/opp_bin; the Python MatchRunner
            # reads cand_path/opp_path). Best-effort — absent until the gate path exports .bin.
            cand_bin = _fetch_net(server, job.get("candidate_bin_sha", ""), gate_dir, timeout, ".bin")
            opp_bin = _fetch_net(server, job.get("opponent_bin_sha", ""), gate_dir, timeout, ".bin")
            if cand_bin is not None:
                job["cand_bin"] = str(cand_bin)
            if opp_bin is not None:
                job["opp_bin"] = str(opp_bin)
        # (a train job is queued verbatim — the worker self-plays with its synced weights.pt)
        tmp = jobs_dir / f"{seq}.json.tmp"
        try:
            tmp.write_text(json.dumps(job))
            tmp.replace(jobs_dir / f"{seq}.json")
        except OSError as e:
            log.warning("job queue write failed: %s", e)
            break
        seq += 1
    return seq


def _ship_match_results(server: str, match_out: Path, timeout: float) -> int:
    """POST each finished gate outcome (match_out/<seq>.json, written atomically by a worker) to
    /match_result, then delete it. The server acks (200) and drops outcomes for a closed gate, so
    a late result is shipped-then-discarded, never retried forever. Returns the count shipped."""
    if not match_out.exists():
        return 0
    shipped = 0
    for f in sorted(match_out.glob("*.json")):
        try:
            body = f.read_bytes()
        except OSError:
            continue
        try:
            _post(f"{server}/match_result", body, timeout, "application/json")
        except (urllib.error.URLError, OSError) as e:
            log.warning("match_result POST failed (retry next tick): %r", e)
            break  # server down — keep results for retry
        try:
            f.unlink()
        except OSError:
            pass
        shipped += 1
    return shipped


def main() -> int:
    setup_logging()
    p = argparse.ArgumentParser(description="Fleet client: lc0 orchestrator over HTTP (owns cc_selfplay).")
    p.add_argument("--server", required=True, help="e.g. http://192.168.1.50:8000")
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Self-play run-dir (shared with the cc_selfplay engines): "
                        "weights.{pt,bin} are written here, games are read from buffer/.")
    p.add_argument("--poll-seconds", type=float, default=15.0)
    p.add_argument("--min-age", type=float, default=2.0,
                   help="Only upload pkls older than this (s) so their .meta has flushed.")
    p.add_argument("--queue-depth", type=int, default=8,
                   help="Target number of UNCLAIMED jobs to keep in run-dir/jobs/ for the owned "
                        "workers (lc0 client-drives-each-game). Keep >= --workers so a worker "
                        "never idles between polls; higher = more gate-switch lag. Self-play "
                        "games take minutes, so a small multiple of the worker count is plenty.")
    p.add_argument("--timeout", type=float, default=30.0, help="per-request HTTP timeout (s)")
    p.add_argument("--client-id", default="",
                   help="fleet liveness id sent as X-Client-Id (default: this box's hostname)")
    p.add_argument("--bind-interface", default="",
                   help="pin outbound HTTP to this interface via macOS IP_BOUND_IF (e.g. en0) — "
                        "use where a VPN scopes sockets onto a tunnel and the LAN server becomes "
                        "unreachable; empty = default routing.")
    p.add_argument("--spawn-engines", action="store_true",
                   help="Own N cc_selfplay --jobs-local ENGINE procs (the lc0 engine binary). "
                        "Each claims jobs from run-dir/jobs/ and plays off run-dir/weights.bin. "
                        "Without it the client is a pure orchestrator (uploads/ships only).")
    p.add_argument("--engine-binary", default="",
                   help="Path to the cc_selfplay executable (default: engine/cpp/build/cc_selfplay "
                        "relative to this client).")
    p.add_argument("--engine-workers", type=int, default=os.cpu_count() or 4,
                   help="--spawn-engines: number of cc_selfplay procs (default: ncpu).")
    p.add_argument("--engine-worker-id-base", type=int, default=0,
                   help="--spawn-engines: first --worker-id (distinct per box, like the Python "
                        "worker-id-base; engine i gets base+i).")
    p.add_argument("--engine-seed", type=int, default=0,
                   help="--spawn-engines: base RNG seed (engine i gets seed+i).")
    p.add_argument("--engine-batch-size", type=int, default=1,
                   help="--spawn-engines: each engine claims this many train jobs and plays them "
                        "concurrently through one shared batched trunk (Phase 6f). 1 = serial (the "
                        "default). On a GPU box pair this with --engine-gpu and --engine-workers 1.")
    p.add_argument("--engine-gpu", action="store_true",
                   help="--spawn-engines: engines batch self-play on the GPU (Metal on Apple). Only "
                        "meaningful with --engine-batch-size >1; falls back to the CPU batched trunk "
                        "if no GPU. ~2x self-play vs the CPU path on this Mac (more on a CUDA box).")
    p.add_argument("--update-cmd", default="",
                   help="Shell command that pulls + rebuilds this box's code when the server "
                        "advertises a newer version (e.g. 'cd ~/chessckers && git pull --ff-only "
                        "&& cd engine && PATH=.venv/bin:$PATH cpp/build.sh'). On success the "
                        "client re-execs onto the fresh code. Empty = warn on drift only.")
    args = p.parse_args()

    server = args.server.rstrip("/")
    bound_idx = None  # the iface index the bound opener cached at startup (for drift checks in _deep_diag)
    if args.bind_interface:
        global _opener
        try:
            _opener, bound_idx = _build_bound_opener(args.bind_interface)
            log.info("pinned outbound HTTP to interface %s (IP_BOUND_IF, idx=%s)", args.bind_interface, bound_idx)
        except OSError as e:
            log.warning("could not bind to %s (%s) — using default routing", args.bind_interface, e)
    client_id = args.client_id or socket.gethostname()
    client_version = _git_version()
    # per-tick heartbeat headers: id + code version (server tracks last-seen + version)
    hb = {"X-Client-Id": client_id, "X-Client-Version": client_version}
    run_dir = args.run_dir.resolve()
    weights = run_dir / "weights.pt"
    weights_bin = run_dir / "weights.bin"  # C++-loadable twin for the cc_selfplay engine
    buffer = run_dir / "buffer"
    stop_path = run_dir / "STOP"
    jobs_dir = run_dir / "jobs"        # server-assigned jobs the workers claim (lc0 client-drives)
    match_out = run_dir / "match_out"  # gate outcomes the workers drop for us to POST /match_result
    gate_dir = run_dir / "_gate"       # content-addressed cache of fetched gate nets (+ .bin exports)
    buffer.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    match_out.mkdir(parents=True, exist_ok=True)

    log.info("fleet client up: server=%s run-dir=%s poll=%.0fs id=%s version=%s",
             server, run_dir, args.poll_seconds, client_id, client_version)
    # The engine pool (Phase 3B-3): N cc_selfplay --jobs-local procs the orchestrator owns,
    # spawned once weights.bin lands. Built here (no procs yet); .ensure()d in step 4.
    engine_pool = None
    if args.spawn_engines:
        binary = args.engine_binary or str(
            Path(__file__).resolve().parent.parent / "cpp" / "build" / "cc_selfplay")
        engine_pool = _EnginePool(
            binary=binary, run_dir=run_dir, n=args.engine_workers,
            worker_id_base=args.engine_worker_id_base, seed=args.engine_seed,
            machine=os.environ.get("MACHINE", "unknown"), log_dir=run_dir,
            batch_size=args.engine_batch_size, use_gpu=args.engine_gpu)
        log.info("owning engines (lc0 client-drives, job-driven): %d x cc_selfplay --jobs-local "
                 "(%s) | worker-id-base=%d queue-depth=%d batch-size=%d gpu=%s", args.engine_workers,
                 binary, args.engine_worker_id_base, args.queue_depth, args.engine_batch_size,
                 args.engine_gpu)
    # A box plays gate games iff it owns the engine pool (cc_selfplay plays match jobs
    # natively); a pure orchestrator declines match jobs.
    can_match = args.spawn_engines
    last_version = ""        # legacy /version fallback tracking (servers without X-Network-Sha)
    have_sha = ""            # content sha of the net currently materialized at weights.pt
    have_bin = ""            # content sha of the .bin twin materialized at weights.bin (cc_selfplay)
    total_up = 0
    total_gate = 0           # gate outcomes shipped to /match_result
    engines_started = False  # reset the job queue once, on the pool's first spawn
    job_seq = 0              # monotonic job-file sequence (unique <seq>.json names in jobs/)
    update_cmd = args.update_cmd
    update_backoff: dict = {}  # sha -> last attempt time, so a self-update isn't retried every tick
    # Reachability timeline (the intermittent-failure signal): count consecutive control
    # failures so a flap logs its onset, duration on recovery, and a periodic "still alive".
    consecutive_fail = 0
    first_fail_t = 0.0
    last_heartbeat_t = 0.0
    last_deep_t = 0.0
    while True:
        try:
            now = time.time()
            # 0. engine liveness for this tick's heartbeat (so the server can flag a
            #    zombie box: client heartbeating but its engines dead). off = not yet
            #    spawned (waiting on weights.bin); up/down = pool all-alive / any-exited.
            if args.spawn_engines:
                ws = engine_pool.status()
                hb_tick = {**hb, "X-Client-Workers": ws}
            else:
                ws = "n/a"
                hb_tick = dict(hb)  # fresh copy so the per-tick net header never mutates `hb`
            if have_sha:  # report the net we're running -> /status fleet net-consistency view
                hb_tick["X-Client-Net"] = have_sha
            # 1. control (also the per-tick liveness heartbeat via the X-Client-Id header).
            #    /control's X-Network-Sha header carries the current net's content address, so
            #    step 2 syncs the net off this same request (None = control failed / legacy server).
            want_sha = None
            want_bin = None  # X-Network-Bin-Sha: the .bin twin's content address (cc_selfplay)
            try:
                cbody, chdrs = _get2(f"{server}/control", args.timeout, hb_tick)
                control = cbody.decode().strip()
                control_ok = True
                want_sha = chdrs.get("X-Network-Sha")
                want_bin = chdrs.get("X-Network-Bin-Sha")
            except (urllib.error.URLError, OSError) as e:
                control = "RUN"  # server unreachable — keep self-playing on current weights
                control_ok = False
                consecutive_fail += 1
                if consecutive_fail == 1:
                    first_fail_t = now
                # One-line egress snapshot every failing tick; the full kitchen-sink _deep_diag
                # on the FIRST failing tick of an outage and every DEEP_DIAG_S thereafter.
                log.warning("control GET failed (heartbeat dropped; %d in a row): %s | %s",
                            consecutive_fail, _err_detail(e), _net_diag(server, args.bind_interface))
                if consecutive_fail == 1 or now - last_deep_t >= DEEP_DIAG_S:
                    last_deep_t = now
                    log.warning("deep-diag @ unreachable:\n    %s",
                                _deep_diag(server, args.bind_interface, bound_idx))
            # Reachability transitions: log recovery (with how long we were down) the first
            # healthy tick after a flap; otherwise emit a periodic "still alive" health line.
            if control_ok:
                if consecutive_fail:
                    log.info("recovered: control reachable after %d failed tick(s) (~%.0fs down)",
                             consecutive_fail, now - first_fail_t)
                    consecutive_fail = 0
                elif now - last_heartbeat_t >= HEARTBEAT_S:
                    last_heartbeat_t = now
                    log.info("health: control=%s net=%s workers=%s uploaded=%d",
                             control, (have_sha[:12] if have_sha else last_version) or "none", ws, total_up)
            if control == "STOP":
                log.info("server signaled STOP -> stopping local engines + exiting")
                try:
                    stop_path.touch()
                except OSError:
                    pass
                if engine_pool is not None:
                    engine_pool.stop(timeout=30)
                break
            # 1b. code self-update (lc0-style): re-exec onto the trainer host's code if
            #     we booted on an older sha. Backoff so it self-corrects if the code lands
            #     here just after the server starts advertising it.
            try:
                want = _get(f"{server}/client-version", args.timeout, hb_tick).decode().strip()
            except (urllib.error.URLError, OSError) as e:
                want = ""
                log.warning("client-version GET failed: %r", e)
            if (want and want not in ("unknown", client_version)
                    and not _head_contains(want)  # never DOWNGRADE: server briefly behind a fresh client
                    and now - update_backoff.get(want, 0.0) > UPDATE_RETRY_S):
                update_backoff[want] = now
                if not update_cmd:
                    log.warning("code drift: server on %s, this box booted on %s "
                                "(no --update-cmd; update manually)", want, client_version)
                else:
                    _self_update(want, client_version, update_cmd, run_dir, stop_path,
                                 engine_pool)
            # 2. net — content-addressed sync off /control's X-Network-Sha (preferred), or the
            #    legacy /version+/weights poll for a server that doesn't advertise the sha.
            #    Skipped when control failed (server unreachable — nothing to pull this tick).
            if control_ok:
                if want_sha is not None:
                    have_sha = _pull_net_by_sha(server, weights, want_sha, have_sha, args.timeout)
                else:
                    last_version = _pull_weights_if_new(server, weights, last_version, args.timeout)
                # The .bin twin (cc_selfplay engine hot-reloads weights.bin). Additive: '' until
                # the server publishes a .bin, and a Python-worker box simply ignores the file.
                if want_bin:
                    have_bin = _pull_sha_to(server, weights_bin, want_bin, have_bin, args.timeout, "bin")
            # 3. finished self-play games (train job output): upload + drop locally.
            n = _upload_games(server, buffer, args.min_age, args.timeout)
            if n:
                total_up += n
                log.info("uploaded %d game(s) <%s> (total %d)", n, client_id, total_up)
            # 3b. finished gate outcomes (match job output): POST /match_result + drop locally.
            g = _ship_match_results(server, match_out, args.timeout)
            if g:
                total_gate += g
                log.info("shipped %d gate result(s) <%s> (total %d)", g, client_id, total_gate)
            # 4. own the cc_selfplay engine pool (lc0 client-owns-engine). Engines need the
            #    C++-loadable net; spawn once weights.bin has landed, then keep the pool full
            #    (dead procs restart individually). Reset the shared queue ONCE on first spawn
            #    so a fresh pool never claims a stale job; a single engine restart does NOT
            #    reset (it would disrupt the other N-1's in-flight claims). STOP is handled
            #    above, so reaching here always means the pool should be kept up.
            if args.spawn_engines and weights_bin.exists():
                if not engines_started:
                    _reset_queue(jobs_dir, match_out)
                    engines_started = True
                engine_pool.ensure()
            # 5. lc0 client-drives-each-game: keep the owned engines fed with server-assigned
            #    jobs. While the queue is below --queue-depth, POST /next_game and queue the
            #    reply for an engine to claim (a `train` job -> a self-play game; a `match` job
            #    -> a gate game, the two nets pre-fetched by sha). The engines claim + play; their
            #    outputs ship in steps 3/3b. Skipped when control failed (server unreachable) or
            #    the pool isn't up yet (nothing to claim the jobs).
            pool_up = engine_pool is not None and engine_pool.any_alive()
            if control_ok and pool_up:
                job_seq = _mint_jobs(server, jobs_dir, gate_dir, args.queue_depth, job_seq,
                                     can_match, args.timeout, hb_tick)
        except Exception as e:  # noqa: BLE001 — unattended box: never die on an unexpected tick error
            log.warning("tick error (continuing): %s", e)
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
