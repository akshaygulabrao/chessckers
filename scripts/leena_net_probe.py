#!/usr/bin/env python3
"""leena_net_probe — rigorously test leena's reachability to the fleet server,
broken down by the network interface the traffic actually leaves on.

Run this ON LEENA. It answers one question: from which of leena's interfaces can a
real TCP+HTTP connection to the server be established? It tests three layers so we can
tell a *link* problem ("no route on this NIC") apart from a *per-process scoping*
problem (a VPN / Network Extension hijacking this process's flows regardless of the
socket's bound interface — the failure mode we saw, where even IP_BOUND_IF=en0 fails).

Layers, per interface:
  1. ROUTE   — what `route get <server>` says the kernel would pick (no traffic sent).
  2. BIND    — open a TCP socket pinned to that interface via IP_BOUND_IF (socket opt 25,
               the exact pin fleet_client --bind-interface uses) and connect.
  3. HTTP    — over that pinned socket, GET /control and show the reply (b'RUN' = healthy).

It also runs an UNPINNED (default-route) HTTP GET as the baseline a fresh process gets.

Usage (on leena):
  python3 scripts/leena_net_probe.py                       # server defaults to .107:8000
  python3 scripts/leena_net_probe.py http://192.168.68.107:8000
  REPEAT=20 python3 scripts/leena_net_probe.py             # loop 20x (catch intermittency)
"""
from __future__ import annotations  # lazy annotations -> runs on leena's system py<3.10 too

import os
import socket
import struct
import subprocess
import sys
import time
from urllib.parse import urlparse

IP_BOUND_IF = 25  # macOS socket option: pin a socket to a named interface


def server_from_argv() -> str:
    for a in sys.argv[1:]:
        if a.startswith("http"):
            return a.rstrip("/")
    return "http://192.168.68.107:8000"


def list_interfaces() -> list[str]:
    """Every UP interface name, in kernel order (en0, en1, utun*, bridge0, ...)."""
    out = subprocess.run(["ifconfig", "-l", "-u"], capture_output=True, text=True).stdout
    names = out.split()
    # lo0 is useless for reaching a LAN peer; keep everything else.
    return [n for n in names if n != "lo0"]


def iface_ipv4(ifname: str) -> str | None:
    out = subprocess.run(["ifconfig", ifname], capture_output=True, text=True).stdout
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("inet ") and "inet6" not in line:
            return line.split()[1]
    return None


def route_get(host: str) -> str:
    """Which interface the kernel would route to host on, by default (no traffic)."""
    out = subprocess.run(["route", "-n", "get", host], capture_output=True, text=True).stdout
    dev = gw = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("interface:"):
            dev = s.split(":", 1)[1].strip()
        elif s.startswith("gateway:"):
            gw = s.split(":", 1)[1].strip()
    return f"dev={dev or '?'} gw={gw or '-'}"


def try_pinned(host: str, port: int, ifname: str, timeout: float = 4.0):
    """(ok, detail) for a TCP connect pinned to ifname via IP_BOUND_IF."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        idx = socket.if_nametoindex(ifname)
        s.setsockopt(socket.IPPROTO_IP, IP_BOUND_IF, struct.pack("I", idx))
    except OSError as e:
        s.close()
        return False, f"setsockopt(IP_BOUND_IF {ifname}) failed: {e!r}"
    try:
        s.connect((host, port))
        local = s.getsockname()  # the source addr the kernel actually used
        return True, f"connected src={local[0]}:{local[1]}"
    except OSError as e:
        return False, f"connect failed: errno={e.errno} {e.strerror}"
    finally:
        s.close()


def http_get_pinned(host: str, port: int, path: str, ifname: str | None, timeout: float = 4.0):
    """Raw HTTP/1.1 GET over a (optionally pinned) socket; returns (ok, detail)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    if ifname:
        try:
            idx = socket.if_nametoindex(ifname)
            s.setsockopt(socket.IPPROTO_IP, IP_BOUND_IF, struct.pack("I", idx))
        except OSError as e:
            s.close()
            return False, f"pin {ifname} failed: {e!r}"
    try:
        s.connect((host, port))
        req = (
            f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
            "X-Client-Id: leena-probe\r\nConnection: close\r\n\r\n"
        )
        s.sendall(req.encode())
        buf = b""
        while len(buf) < 4096:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
        status = buf.split(b"\r\n", 1)[0].decode(errors="replace") if buf else "(no data)"
        body = buf.split(b"\r\n\r\n", 1)[1] if b"\r\n\r\n" in buf else b""
        return True, f"{status}  body={body[:40]!r}"
    except OSError as e:
        return False, f"errno={e.errno} {e.strerror}"
    finally:
        s.close()


def run_once(server: str):
    u = urlparse(server)
    host = socket.gethostbyname(u.hostname)
    port = u.port or 80
    print(f"== target {server}  ->  {host}:{port} ==")
    print(f"   default route to host: {route_get(host)}")

    # Baseline: unpinned (default route) HTTP GET — what a fresh process gets.
    ok, detail = http_get_pinned(host, port, "/control", None)
    print(f"   [UNPINNED default ] HTTP /control: {'OK ' if ok else 'FAIL'} {detail}")
    print("   per-interface:")
    print(f"   {'iface':<10} {'ipv4':<16} {'route':<26} {'tcp-bind':<34} http")
    for ifname in list_interfaces():
        ip = iface_ipv4(ifname) or "-"
        rt = route_get(host)
        tok, tdetail = try_pinned(host, port, ifname)
        if tok:
            hok, hdetail = http_get_pinned(host, port, "/control", ifname)
            http = ("OK  " if hok else "FAIL ") + hdetail
        else:
            http = "(skipped: no TCP)"
        print(f"   {ifname:<10} {ip:<16} {rt:<26} {('OK '+tdetail) if tok else ('FAIL '+tdetail):<34} {http}")
    print()


def main():
    server = server_from_argv()
    repeat = int(os.environ.get("REPEAT", "1"))
    for i in range(repeat):
        if repeat > 1:
            print(f"--- iteration {i+1}/{repeat}  t={time.strftime('%H:%M:%S')} ---")
        run_once(server)
        if i + 1 < repeat:
            time.sleep(3)


if __name__ == "__main__":
    main()
