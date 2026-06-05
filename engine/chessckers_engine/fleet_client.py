"""Fleet client — lc0-style self-play client sync bridge.

Runs on every self-play box (the local machine over loopback AND leena / future
volunteers over the LAN). The actual game generation is the existing
`selfplay_workers_only` process writing pkls into `--run-dir/buffer` and
mtime-polling `--run-dir/weights.pt`; this bridge is the only thing that talks to
the server, so a volunteer needs no inbound SSH — just outbound HTTP. It replaces
the rsync `leena_sync.sh` sidecar.

With `--spawn-workers` it also OWNS that worker process (lc0's client-owns-engine
model): it launches `selfplay_workers_only` once weights land, restarts it if it
dies, and heartbeats its liveness to the server — so a zombie box (client up,
workers dead, producing nothing) shows up in /status instead of silently lying.

Each tick (`--poll-seconds`):
  1. GET /control (carrying `X-Client-Id` + `X-Client-Version` heartbeat headers so
     the server can list live boxes AND their code version in /status — a box on a
     stale commit is then visible, not a silent runtime crash) — if STOP, signal the
     local workers (touch run-dir/STOP) and exit. A tick that raises an unexpected
     error is logged and retried next poll — the loop never dies (an unattended
     volunteer box must outlive transient faults).
  1b. GET /client-version — the trainer host's code sha. With --update-cmd set, a
     mismatch against the sha this client booted on triggers a pull + native rebuild
     and an os.execv re-exec onto the fresh code (retried with backoff so it self-
     corrects even if the code lands here just after the server advertises it). Without
     --update-cmd the drift is only logged, so a stale box stays visible in /status.
  2. Net sync (content-addressed, lc0 get_network): /control's response carries an
     X-Network-Sha header (the current net's sha256); on a change, GET /get_network?sha=
     and write run-dir/weights.pt atomically (the workers hot-reload on their own mtime
     poll). The client gets back exactly the bytes it asked for, keyed by hash, not an
     opaque version. Falls back to the legacy /version+/weights poll against a server
     that doesn't advertise the sha. Fires once per trainer ITERATION / promotion.
  2b. GET /selfplay — mirror the server's canonical self-play params into
     run-dir/selfplay.json (only on a content change). The workers re-read it each
     game, so the whole fleet self-plays with the SAME, operator-tunable params and
     can be annealed mid-run without a relaunch.
  3. Upload finished games: each `*.pkl` older than --min-age is POSTed with its
     `.meta` in ONE multipart /upload_game request (server lands meta-before-pkl, so
     the trainer never sees a meta-less game), then deleted locally — each game
     uploaded exactly once. (The .pkl is now a gzipped-JSON `ccz` chunk, not a
     pickle; the client moves the bytes opaquely.)
  4. (--spawn-workers) Supervise the self-play worker subprocess: spawn it once
     weights.pt has landed, restart it on unexpected exit, and heartbeat its state
     (X-Client-Workers: up/down/off) on steps 1/2b so /status can flag a zombie box.
  5. Keep-best gate (lc0 POST /next_game): ask for a job. A `match` job (a gate is open)
     means PAUSE our owned workers (touch run-dir/PAUSE so they idle instead of contending
     for CPU), fetch the two nets by content address (/get_network?sha=), play ONE gate game
     in-process, and POST /match_result. A `train` job (no gate) resumes the workers. The
     heavy player (fleet_match) is imported lazily on the first match job, so a box without
     torch/the native ext stays self-play-only.

The client is stdlib only (urllib) — no requests/aiohttp dep, so it runs on a
bare volunteer venv. Step 4 shells out to the worker (never imports it), so the
client never pulls in torch / the move-gen ext.

Run (on a self-play box):

    python -m chessckers_engine.fleet_client \\
      --server http://192.168.1.50:8000 --run-dir ~/chessckers/run --poll-seconds 15 \\
      --spawn-workers -- --workers 4 --native --device cpu \\
      --d-hidden 256 --c-filters 96 --n-blocks 4 --worker-id-base 300

Everything after `--` is the worker command (selfplay_workers_only's flags); the
client injects --run-dir/--weights, so the launcher only passes box-specific config.
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


def _split_worker_argv(argv: list) -> tuple:
    """Split CLI args at the first literal `--`: everything before is the client's own
    flags, everything after is the worker command (for --spawn-workers). No `--` => no
    worker command."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def _spawn_workers(worker_argv: list, run_dir: Path, weights: Path, log_path: Path):
    """Launch selfplay_workers_only as a supervised child (lc0's client-owns-engine
    model). The client injects --run-dir/--weights (it owns those paths); `worker_argv`
    carries the box-specific config the launcher passed after `--`. Worker stdout/stderr
    go to workers.log so the client's own log stays readable. Shelling out (not
    importing) keeps the client stdlib-only. Returns (Popen, open-file)."""
    cmd = [sys.executable, "-m", "chessckers_engine.selfplay_workers_only",
           "--run-dir", str(run_dir), "--weights", str(weights), *worker_argv]
    f = open(log_path, "a")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    log.info("spawned workers (pid %d) -> %s | args: %s",
             proc.pid, log_path, " ".join(worker_argv) or "(none)")
    return proc, f


def _self_update(want: str, current: str, update_cmd: str, run_dir: Path,
                 stop_path: Path, worker_proc) -> None:
    """Pull + rebuild this box onto code sha `want` via `update_cmd`, then re-exec this
    client onto the fresh code (lc0's client-updates-itself, adapted). On success
    os.execv replaces this process and never returns. On failure (cmd errored, or the
    working tree didn't actually advance to `want`) it logs and returns, leaving the box
    visibly on `current` — the caller's backoff retries later (e.g. once the code lands).
    Owned workers are wound down before the rebuild so a stale .so isn't mid-search."""
    log.warning("code self-update: server on %s, running %s — running update-cmd", want, current)
    if worker_proc is not None:
        try:
            stop_path.touch()
            worker_proc.terminate()
        except OSError:
            pass
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


def _pull_net_by_sha(server: str, weights: Path, want_sha: str | None, have_sha: str,
                     timeout: float) -> str:
    """Content-addressed net sync (lc0 get_network): fetch the net whose sha256 is
    `want_sha` only when it differs from what we last wrote, and materialize it at
    weights.pt (atomic; bumps mtime -> the local workers hot-reload). `want_sha` comes
    from the X-Network-Sha header on /control, so an unchanged net costs nothing. Unlike
    the opaque /version, the client gets back EXACTLY the bytes it asked for (or a 404 if
    the net rotated mid-fetch -> retry next tick). Returns the sha now on disk (unchanged
    on empty want / no-op / fetch failure)."""
    if not want_sha or want_sha == have_sha:
        return have_sha
    try:
        data = _get(f"{server}/get_network?sha={want_sha}", timeout)
    except (urllib.error.URLError, OSError) as e:
        log.warning("net fetch (sha %s) failed: %r", want_sha[:12], e)
        return have_sha  # 404 (rotated) or server blip — retry next tick on the new sha
    tmp = weights.with_suffix(".pt.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, weights)  # atomic; bumps mtime -> local workers hot-reload
    except OSError as e:
        log.warning("weights write failed: %s", e)
        return have_sha
    log.info("pulled net sha %s -> %s (%d KB)", want_sha[:12], weights, len(data) // 1024)
    return want_sha


def _pull_selfplay_if_new(server: str, params_path: Path, last: bytes, timeout: float,
                          headers: dict) -> bytes:
    """Mirror the server's canonical self-play params into the local run-dir. The
    workers re-read this file each game, so a server-side change anneals this box at
    the next game boundary — same mechanism as the weights pull, one level finer.
    Only rewrites on a CONTENT change so an unchanged file's mtime stays put and the
    workers don't needlessly re-parse. Returns the current bytes for tracking."""
    try:
        data = _get(f"{server}/selfplay", timeout, headers)
    except (urllib.error.URLError, OSError) as e:
        log.warning("selfplay poll failed (server down/restarting?): %r", e)
        return last
    if data == last:
        return last
    tmp = params_path.with_suffix(".json.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, params_path)
    except OSError as e:
        log.warning("selfplay params write failed: %s", e)
        return last
    log.info("self-play params updated from server -> %s",
             data.decode("utf-8", "replace").strip())
    return data


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


def _clear_pause(pause_path: Path) -> None:
    """Remove the PAUSE sentinel so our owned self-play workers resume (no-op if absent)."""
    try:
        pause_path.unlink()
    except OSError:
        pass


def _fetch_net(server: str, sha: str, cache_dir: Path, timeout: float) -> Path | None:
    """Fetch a gate net by content address (lc0 get_network) into cache_dir/<sha>.pt,
    skipping the download when it's already there (sha-named -> content-addressed, so a
    champion reused across gates isn't re-pulled). Routed through the interface-bound opener
    so leena's en0 pin is honored. None on empty sha / fetch / write failure -> retry next
    tick (e.g. the net rotated mid-gate -> 404)."""
    if not sha:
        return None
    dest = cache_dir / f"{sha}.pt"
    if dest.exists():
        return dest
    try:
        data = _get(f"{server}/get_network?sha={sha}", timeout)
    except (urllib.error.URLError, OSError) as e:
        log.warning("gate net fetch (sha %s) failed: %r", sha[:12], e)
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".pt.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, dest)
    except OSError as e:
        log.warning("gate net write failed: %s", e)
        return None
    return dest


def _play_gate_job(server: str, job: dict, gate_dir: Path, runner, device: str,
                   timeout: float) -> tuple:
    """Play ONE keep-best gate unit (lc0 `match` job) on this box and POST its outcome.
    The heavy player (torch + the native ext) is imported LAZILY here, so a box without the
    deps disables gating and stays self-play-only. Returns (runner, disabled): `runner` is
    the (possibly just-created) MatchRunner to reuse next tick; `disabled` is True iff the
    engine deps are missing. A net not yet fetchable, or a single bad game, is logged and
    skipped (retried next tick) — gating never kills the client."""
    if runner is None:
        try:
            from chessckers_engine.fleet_match import MatchRunner
        except Exception as e:  # noqa: BLE001 — no torch/ext here: never offer gates again
            log.warning("gate offered but engine deps unavailable (%r) — self-play only", e)
            return None, True
        runner = MatchRunner(gate_dir, device=device)
    cand = _fetch_net(server, job.get("candidate_sha", ""), gate_dir, timeout)
    opp = _fetch_net(server, job.get("opponent_sha", ""), gate_dir, timeout)
    if cand is None or opp is None:
        return runner, False  # net not ready/reachable — retry next tick
    try:
        outcome = runner.play(job, cand, opp)
    except Exception as e:  # noqa: BLE001 — a single bad gate game must not kill the client
        log.warning("gate game failed (%r) — skipping unit", e)
        return runner, False
    try:
        _post_recv(f"{server}/match_result", json.dumps({
            "match_id": job["match_id"], "seed": job["seed"], "opp": job["opponent"],
            "cand_white": job["cand_white"], "outcome": outcome,
        }).encode(), timeout)
    except (urllib.error.URLError, OSError) as e:
        log.warning("match_result POST failed (%r) — outcome dropped", e)
    else:
        log.info("gate game: match %s opp=%s cand_white=%s -> %s",
                 job["match_id"], job["opponent"], job["cand_white"], outcome)
    return runner, False


def main() -> int:
    setup_logging()
    # The worker command (for --spawn-workers) is everything after a literal `--`;
    # split it off before argparse so the client's own flags parse cleanly.
    argv, worker_argv = _split_worker_argv(sys.argv[1:])
    p = argparse.ArgumentParser(description="Fleet client: pull net / push games over HTTP (lc0-style).")
    p.add_argument("--server", required=True, help="e.g. http://192.168.1.50:8000")
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Self-play run-dir (shared with selfplay_workers_only): "
                        "weights.pt is written here, games are read from buffer/.")
    p.add_argument("--poll-seconds", type=float, default=15.0)
    p.add_argument("--min-age", type=float, default=2.0,
                   help="Only upload pkls older than this (s) so their .meta has flushed.")
    p.add_argument("--timeout", type=float, default=30.0, help="per-request HTTP timeout (s)")
    p.add_argument("--client-id", default="",
                   help="fleet liveness id sent as X-Client-Id (default: this box's hostname)")
    p.add_argument("--bind-interface", default="",
                   help="pin outbound HTTP to this interface via macOS IP_BOUND_IF (e.g. en0) — "
                        "use where a VPN scopes sockets onto a tunnel and the LAN server becomes "
                        "unreachable; empty = default routing.")
    p.add_argument("--spawn-workers", action="store_true",
                   help="Own the self-play worker subprocess (lc0-style): launch "
                        "selfplay_workers_only, restart it if it dies, report liveness. "
                        "Worker flags follow a `--` separator; --run-dir/--weights injected.")
    p.add_argument("--update-cmd", default="",
                   help="Shell command that pulls + rebuilds this box's code when the server "
                        "advertises a newer version (e.g. 'cd ~/chessckers && git pull --ff-only "
                        "&& cd engine && PATH=.venv/bin:$PATH cpp/build.sh'). On success the "
                        "client re-execs onto the fresh code. Empty = warn on drift only.")
    args = p.parse_args(argv)

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
    params_path = run_dir / "selfplay.json"
    buffer = run_dir / "buffer"
    stop_path = run_dir / "STOP"
    pause_path = run_dir / "PAUSE"   # touched while we play a gate game -> our owned workers idle
    gate_dir = run_dir / "_gate"     # content-addressed cache of fetched gate nets (+ .bin exports)
    buffer.mkdir(parents=True, exist_ok=True)
    _clear_pause(pause_path)         # drop any PAUSE a previously-crashed client left (workers would hang)

    log.info("fleet client up: server=%s run-dir=%s poll=%.0fs id=%s version=%s",
             server, run_dir, args.poll_seconds, client_id, client_version)
    if args.spawn_workers:
        log.info("owning workers (lc0-style): selfplay_workers_only %s",
                 " ".join(worker_argv) or "(no extra args)")
    last_version = ""        # legacy /version fallback tracking (servers without X-Network-Sha)
    have_sha = ""            # content sha of the net currently materialized at weights.pt
    last_selfplay = b""
    total_up = 0
    worker_proc = None
    worker_log = None
    gate_runner = None        # lazily-created MatchRunner (heavy import deferred to the first match job)
    gate_paused = False       # have we paused our owned workers for an open gate?
    gate_disabled = False     # engine deps missing -> never offer gates on this box
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
            # 0. worker liveness for this tick's heartbeat (so the server can flag a
            #    zombie box: client heartbeating but its workers dead). off = not yet
            #    spawned (waiting on weights); up/down = supervised child alive/exited.
            if args.spawn_workers:
                ws = ("down" if (worker_proc is not None and worker_proc.poll() is not None)
                      else "up" if worker_proc is not None else "off")
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
            try:
                cbody, chdrs = _get2(f"{server}/control", args.timeout, hb_tick)
                control = cbody.decode().strip()
                control_ok = True
                want_sha = chdrs.get("X-Network-Sha")
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
                log.info("server signaled STOP -> stopping local workers + exiting")
                try:
                    stop_path.touch()
                except OSError:
                    pass
                _clear_pause(pause_path)  # don't leave workers blocked on a stale gate PAUSE
                gate_paused = False
                if worker_proc is not None:
                    try:
                        worker_proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        log.warning("workers slow to stop; terminating")
                        worker_proc.terminate()
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
                    _self_update(want, client_version, update_cmd, run_dir, stop_path, worker_proc)
            # 2. net — content-addressed sync off /control's X-Network-Sha (preferred), or the
            #    legacy /version+/weights poll for a server that doesn't advertise the sha.
            #    Skipped when control failed (server unreachable — nothing to pull this tick).
            if control_ok:
                if want_sha is not None:
                    have_sha = _pull_net_by_sha(server, weights, want_sha, have_sha, args.timeout)
                else:
                    last_version = _pull_weights_if_new(server, weights, last_version, args.timeout)
            # 2b. self-play params (server-published; workers live-apply per game)
            last_selfplay = _pull_selfplay_if_new(server, params_path, last_selfplay, args.timeout, hb_tick)
            # 3. games
            n = _upload_games(server, buffer, args.min_age, args.timeout)
            if n:
                total_up += n
                log.info("uploaded %d game(s) <%s> (total %d)", n, client_id, total_up)
            # 4. own the worker subprocess (lc0 client-owns-engine). Spawn once weights
            #    have landed; restart on unexpected exit (self-heal). STOP is handled
            #    above, so reaching here always means a (re)start is wanted.
            if args.spawn_workers:
                if worker_proc is None:
                    if weights.exists():
                        worker_proc, worker_log = _spawn_workers(
                            worker_argv, run_dir, weights, run_dir / "workers.log")
                elif worker_proc.poll() is not None:
                    log.warning("workers exited (rc=%s) — restarting", worker_proc.returncode)
                    if worker_log is not None:
                        worker_log.close()
                    worker_proc, worker_log = _spawn_workers(
                        worker_argv, run_dir, weights, run_dir / "workers.log")
            # 5. keep-best gate (lc0 POST /next_game). A `match` job means a gate is open ->
            #    pause our owned self-play workers (so the in-process gate game doesn't contend
            #    for CPU) and play ONE unit; a `train` job means no gate -> resume them if paused.
            #    The heavy player is imported lazily on the first match job; a box without the
            #    engine deps disables gating and stays self-play-only. Skipped when control failed
            #    (server unreachable) so we don't pause workers on a blip.
            if control_ok and not gate_disabled:
                try:
                    job = json.loads(_post_recv(f"{server}/next_game", b"", args.timeout, hb_tick))
                except (urllib.error.URLError, OSError, ValueError) as e:
                    job = None
                    log.debug("next_game poll failed: %r", e)
                if job is not None and job.get("type") == "match":
                    if args.spawn_workers and not gate_paused:
                        try:
                            pause_path.touch()
                        except OSError:
                            pass
                        gate_paused = True
                        log.info("gate open (match %s) — pausing self-play workers to contribute gate games",
                                 job.get("match_id"))
                    # Gate games run on CPU (the arena's gate is CPU; avoids MPS contention with
                    # the trainer on the local box and matches leena's CPU self-play).
                    gate_runner, gate_disabled = _play_gate_job(
                        server, job, gate_dir, gate_runner, "cpu", args.timeout)
                    if gate_disabled and gate_paused:  # deps missing after all -> let workers run
                        _clear_pause(pause_path)
                        gate_paused = False
                elif job is not None and gate_paused:  # train job: gate closed -> resume workers
                    _clear_pause(pause_path)
                    gate_paused = False
                    log.info("gate closed — resuming self-play workers")
        except Exception as e:  # noqa: BLE001 — unattended box: never die on an unexpected tick error
            log.warning("tick error (continuing): %s", e)
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
