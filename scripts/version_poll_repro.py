#!/usr/bin/env python3
"""version_poll_repro — reproduce ONLY the fleet_client HTTP poll path (the /version &
/control GETs), completely separate from self-play / torch / the worker subprocess.

Why this exists: the failure ('No route to host', EHOSTUNREACH) shows up ONLY inside the
long-running fleet_client; standalone ping / preflight / curl never fail. So it isn't the
link — it's something in the process's own HTTP path. The leading suspect is
fleet_client._build_bound_opener, which caches `if_nametoindex(en0)` ONCE at startup and
reuses that one index for every pinned socket. A fresh standalone probe re-reads the index
each time (so it always works); a long-lived process holding a STALE index would get
EHOSTUNREACH on every pinned connect while ping/curl keep working.

This script isolates that. Each tick it fires the SAME /version (and /control) GET three ways
and prints them side by side:
  PIN(cached)  — opener built ONCE at startup, exactly like fleet_client (the suspect)
  PIN(fresh)   — opener rebuilt this tick, re-reading if_nametoindex (control for caching)
  UNPIN        — default routing, no IP_BOUND_IF (what ping/curl effectively do)

Read the verdict from the divergence on a bad tick:
  cached FAIL, fresh OK,  unpin OK  -> STALE CACHED INDEX  (the one-time caching is the bug)
  cached FAIL, fresh FAIL, unpin OK -> pinning to en0 itself is the problem, not caching
  all FAIL                          -> genuinely the link (per you: shouldn't happen here)
On a bad tick it also dumps route dev + cached-vs-live en0 index so drift is visible.

Run ON LEENA (stdlib only; system python3 or the venv both work):
  python3 scripts/version_poll_repro.py http://192.168.68.107:8000 en0
  POLL=5 python3 scripts/version_poll_repro.py            # 5s cadence (default), run forever
  POLL=15 N=40 python3 scripts/version_poll_repro.py ...  # 15s like the client, 40 ticks then stop
"""
from __future__ import annotations

import errno
import http.client
import os
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

IP_BOUND_IF = 25  # macOS: pin a socket to a named interface (the --bind-interface mechanism)


def build_bound_opener(ifname):
    """EXACT copy of fleet_client._build_bound_opener — caches the iface index ONCE (closure)."""
    idx = socket.if_nametoindex(ifname)

    class _BoundConn(http.client.HTTPConnection):
        def connect(self):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_IP, IP_BOUND_IF, struct.pack("I", idx))
            if self.timeout is not None and self.timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                s.settimeout(self.timeout)
            s.connect((self.host, self.port))
            self.sock = s

    class _BoundHandler(urllib.request.HTTPHandler):
        def http_open(self, req):
            return self.do_open(_BoundConn, req)

    return urllib.request.build_opener(_BoundHandler), idx


def do_get(opener, url, timeout=10.0):
    """(ok, detail). opener=None -> default routing (unpinned)."""
    req = urllib.request.Request(url, headers={"X-Client-Id": "repro"})
    t0 = time.time()
    try:
        resp = opener.open(req, timeout=timeout) if opener else urllib.request.urlopen(req, timeout=timeout)
        with resp as r:
            body = r.read()
            status = r.status
        return True, f"{status} {body[:12]!r} {(time.time() - t0) * 1000:.0f}ms"
    except (urllib.error.URLError, OSError) as e:
        inner = getattr(e, "reason", None) or e
        errno = getattr(inner, "errno", "?")
        return False, f"FAIL errno={errno} ({inner})"


def sh(cmd, timeout=3.0):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(err {e})"


def raw_connect(host, port, ifname, timeout=2.0):
    """Fresh in-process TCP connect (optionally IP_BOUND_IF-pinned); errno + chosen local addr."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        if ifname:
            s.setsockopt(socket.IPPROTO_IP, IP_BOUND_IF, struct.pack("I", socket.if_nametoindex(ifname)))
        t0 = time.time()
        s.connect((host, port))
        return "OK src=%s:%d %.0fms" % (*s.getsockname(), (time.time() - t0) * 1000)
    except OSError as e:
        return f"FAIL {errno.errorcode.get(e.errno, '?')}({e.errno}) {e.strerror}"
    finally:
        s.close()


def deep_diag(host, port, ifname, cached_idx):
    """Kitchen-sink egress snapshot for a 'No route to host' (EHOSTUNREACH): ARP/neighbor entry
    (the crux), host route flags, bound-iface link state, AWDL radio, TCP socket census
    (TIME_WAIT / ephemeral pile-up), memory+swap pressure, live-vs-cached iface index, and fresh
    pinned+unpinned connects. Best-effort, never raises. Mirrors fleet_client._deep_diag."""
    L = []
    L.append("arp:     " + (sh(["arp", "-n", host]) or "(no entry)"))
    rt = sh(["route", "-n", "get", host])
    L.append("route:   " + " ".join(s.strip() for s in rt.splitlines()
             if any(k in s for k in ("interface:", "gateway:", "flags:", "expire:"))))
    eg = sh(["ifconfig", ifname])
    head = eg.splitlines()[0] if eg else ""
    inet = next((s.strip() for s in eg.splitlines() if s.strip().startswith("inet ")), "(no inet)")
    status = next((s.strip() for s in eg.splitlines() if "status:" in s), "")
    L.append(f"{ifname}:    {head.split('mtu')[0].strip()} | {inet} | {status}")
    aw = sh(["ifconfig", "awdl0"])
    awf = aw.split("flags=")[1].split("<")[0] if "flags=" in aw else "?"
    aws = next((s.strip() for s in aw.splitlines() if "status:" in s), "")
    L.append(f"awdl0:   flags={awf} {aws}")
    ns = sh(["netstat", "-an", "-p", "tcp"])
    states = {}
    nsrv = 0
    for r in (s.split() for s in ns.splitlines()):
        if len(r) >= 6 and r[-1].isupper():
            states[r[-1]] = states.get(r[-1], 0) + 1
        if any(f"{host}.{port}" in tok for tok in r):
            nsrv += 1
    L.append(f"tcp:     {states}  conns->{host}.{port}={nsrv}")
    L.append("mem:     pressure_lvl=" + sh(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]) +
             "  " + sh(["sysctl", "-n", "vm.swapusage"]))
    try:
        live = socket.if_nametoindex(ifname)
        L.append(f"idx:     {ifname} live={live} cached={cached_idx}" + ("  <-- DRIFTED" if live != cached_idx else ""))
    except OSError as e:
        L.append(f"idx:     err {e}")
    L.append(f"connect unpin:     {raw_connect(host, port, None)}")
    L.append(f"connect pin {ifname}: {raw_connect(host, port, ifname)}")
    return "\n            ".join(L)


def main():
    server = next((a for a in sys.argv[1:] if a.startswith("http")), "http://192.168.68.107:8000").rstrip("/")
    ifname = next((a for a in sys.argv[1:] if not a.startswith("http")), "en0")
    poll = float(os.environ.get("POLL", "5"))
    n = int(os.environ.get("N", "0"))  # 0 = run forever
    u = urlparse(server)
    host = socket.gethostbyname(u.hostname)
    port = u.port or 80

    cached_opener, cached_idx = build_bound_opener(ifname)
    print(f"repro: server={server} pin={ifname} cached_idx={cached_idx} poll={poll}s n={n or 'forever'}",
          flush=True)

    i = fails = 0
    while n == 0 or i < n:
        i += 1
        ts = time.strftime("%H:%M:%S")
        cok, cd = do_get(cached_opener, server + "/version")          # the suspect path
        fresh_opener, _ = build_bound_opener(ifname)                  # fresh index, this tick
        fok, fd = do_get(fresh_opener, server + "/version")
        uok, ud = do_get(None, server + "/version")                   # unpinned baseline
        bad = not (cok and fok and uok)
        flag = ">>> ANOMALY " if bad else ""
        line = (f"{ts} #{i} {flag}PIN(cached)={'OK' if cok else cd}  "
                f"PIN(fresh)={'OK' if fok else fd}  UNPIN={'OK' if uok else ud}")
        if bad:
            fails += 1
            line += "\n            " + deep_diag(host, port, ifname, cached_idx)
        print(line, flush=True)
        if n == 0 or i < n:
            time.sleep(poll)
    print(f"done: {fails} anomalous tick(s) of {i}", flush=True)


if __name__ == "__main__":
    main()
