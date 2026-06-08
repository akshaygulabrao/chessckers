#!/usr/bin/env bash
# Leena HTTP-fleet launcher (lc0-style, client-owns-engine). Starts ONE process: the
# fleet_client, which OWNS the native cc_selfplay engines. It pulls the net + the canonical
# self-play params, spawns N cc_selfplay --jobs-local procs once weights land, restarts them
# if they die, uploads finished games, contributes keep-best GATE games, self-updates on a new
# server code version, and reports engine liveness to the server. Shared shape (arch,
# max-plies, seed mix, sims fallback) comes from scripts/fleet.env so leena CANNOT drift
# from local; only box-specific bits (LAN server, client-id, worker-id-base 300, caffeinate,
# self-update) live here. This is the SAME client path as launch_local.sh (loopback).
#
# Deploy: from local  `git push leena main`  then on leena
#   cd ~/chessckers && git pull --ff-only && bash scripts/launch_fleet_leena.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/scripts/fleet.env"
SERVER="${SERVER:-http://192.168.68.107:8000}"
ENG="$REPO_ROOT/engine"
PY="$ENG/.venv/bin/python"
RUN="$ENG/weights/run-local"          # leena's own client run-dir (mirrors the local client)
SEED_MIX="$REPO_ROOT/scripts/seed_mix.txt"
cd "$ENG" || exit 1

fleet_export_env
export MACHINE=leena
# Same seed mix as local (scripts/seed_mix.txt) -> no curriculum drift between boxes.
export CHESSCKERS_START_FEN="$(fleet_seed_fens "$SEED_MIX")"

mkdir -p "$RUN/buffer"
rm -f "$RUN/STOP" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Visibility (observe-only — never blocks the launch). Before the long-lived client
# loop starts we (a) snapshot leena's network/power state and (b) run a one-shot
# connectivity preflight to the server — exactly the GET /control the client makes,
# both unpinned and pinned to en0. A broken box then gives feedback in SECONDS
# instead of "launch, tail the log, and guess". Everything is timestamped and tee'd
# to $DIAG so successive launches can be diff'd when the connection flaps.
# ---------------------------------------------------------------------------
DIAG="$RUN/launch_diag.log"
SERVER_HOST="$(printf '%s' "$SERVER" | sed -E 's#^https?://##; s#[:/].*$##')"
diag(){ printf '%s %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$DIAG"; }

diag "================ launch_fleet_leena $(date '+%F %T') ================"
diag "host=$(hostname)  up=$(uptime | sed -E 's/^.*up //; s/, *[0-9]* users?.*$//')"
diag "git=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')  server=$SERVER (host $SERVER_HOST)"
diag "en0 ipv4=$(ipconfig getifaddr en0 2>/dev/null || echo NONE)"
diag "route->server: $(route -n get "$SERVER_HOST" 2>/dev/null | awk '/interface:/{i=$2}/gateway:/{g=$2}END{print "dev="(i?i:"?")" gw="(g?g:"-")}')"
# VPN / tunnel interfaces present — the documented socket-scoping failure mode where a
# utun captures outbound flows and the LAN server becomes unreachable.
diag "tunnels up: $(ifconfig -l 2>/dev/null | tr ' ' '\n' | grep -E '^(utun|ipsec|ppp)' | paste -sd, - | sed 's/^$/none/')"
command -v scutil >/dev/null 2>&1 && diag "vpn(nc) connected: $(scutil --nc list 2>/dev/null | grep -c '^\* .*Connected')"
# Power — idle-sleep is the #1 way this box goes unreachable; confirm it's on AC and that
# the standalone caffeinate (started below) is actually fighting the configured sleep timers.
diag "power: $(pmset -g batt 2>/dev/null | tail -1 | sed -E 's/^[[:space:]]*//')"
diag "sleep cfg: $(pmset -g 2>/dev/null | grep -E '^[[:space:]]*(sleep|displaysleep|disksleep)[[:space:]]' | tr -s ' \n' '  ' | sed 's/^ //')"

diag "--- connectivity preflight (GET /control — exactly what the client does) ---"
if "$PY" - "$SERVER" en0 <<'PY' 2>&1 | tee -a "$DIAG"
import socket, struct, sys, time
from urllib.parse import urlparse
IP_BOUND_IF = 25  # macOS: pin a socket to a named interface (the --bind-interface mechanism)
server, ifname = sys.argv[1], sys.argv[2]
u = urlparse(server); host = socket.gethostbyname(u.hostname); port = u.port or 80
def probe(pin):
    s = socket.socket(); s.settimeout(4.0)
    try:
        if pin:
            s.setsockopt(socket.IPPROTO_IP, IP_BOUND_IF, struct.pack("I", socket.if_nametoindex(pin)))
        t0 = time.time()
        s.connect((host, port))
        src = s.getsockname()
        s.sendall(f"GET /control HTTP/1.1\r\nHost: {host}:{port}\r\n"
                  f"X-Client-Id: leena-preflight\r\nConnection: close\r\n\r\n".encode())
        buf = b""
        while len(buf) < 4096:
            c = s.recv(4096)
            if not c: break
            buf += c
        dt = (time.time() - t0) * 1000
        line = buf.split(b"\r\n", 1)[0].decode(errors="replace") if buf else "(no data)"
        body = buf.split(b"\r\n\r\n", 1)[1][:16] if b"\r\n\r\n" in buf else b""
        return True, f"OK  {line}  body={body!r}  src={src[0]}:{src[1]}  {dt:.0f}ms"
    except OSError as e:
        return False, f"FAIL errno={e.errno} {e.strerror}"
    finally:
        s.close()
uok, ud = probe(None)
pok, pd = probe(ifname)
print(f"   UNPINNED (default route): {ud}")
print(f"   PINNED   ({ifname} IP_BOUND_IF): {pd}")
sys.exit(0 if pok else 1)
PY
then
  diag "PREFLIGHT OK: en0-pinned GET /control reached $SERVER"
else
  diag "PREFLIGHT FAIL: en0-pinned GET /control did NOT reach $SERVER — launching anyway"
  diag "  (client self-heals when the LAN returns)"
fi

# Keep the Air awake with a STANDALONE detached caffeinate (survives ssh teardown; a
# wrapping one does not). Needs leena on AC power.
pkill -x caffeinate 2>/dev/null || true
nohup caffeinate -ims >/dev/null 2>&1 </dev/null &
CAFF_PID=$!; disown

# Never double-launch. The client owns the engines, so killing it is enough, but also reap any
# stray engines before starting.
pkill -f "chessckers_engine.fleet_client" 2>/dev/null || true
pkill -f "cc_selfplay .*--jobs-local" 2>/dev/null || true
sleep 1

# Native C++ engine (lc0-split cutover, Phase 3B-3): leena (Apple silicon) runs the
# cc_selfplay binary as the engine — ALWAYS rebuild after a code pull (cpp/build.sh builds
# both the .so and cc_selfplay). cmake is a uv-pip wheel in the venv bin, so put it on PATH
# (the venv is never activated here). There is no Python self-play fallback anymore — if the
# build fails, the box can't self-play (the prior code commit's binary stays in place).
CC_SELFPLAY="$ENG/cpp/build/cc_selfplay"
if [ -x cpp/build.sh ]; then
  echo "leena: rebuilding chessckers_cpp + cc_selfplay (cpp/build.sh)…"
  PATH="$ENG/.venv/bin:$PATH" cpp/build.sh > "$RUN/cpp_build.log" 2>&1 \
    || echo "leena: native build FAILED (see run-local/cpp_build.log) — using prior cc_selfplay"
fi
if [ -x "$CC_SELFPLAY" ]; then
  echo "leena: native C++ engine -> cc_selfplay --jobs-local ($FLEET_WORKERS procs)"
else
  echo "leena: ERROR — cc_selfplay not built at $CC_SELFPLAY; cannot self-play"; exit 1
fi

# Self-update command: when the server advertises a newer code sha than this client booted
# on, pull the bare repo into the tree, rebuild the native ext, and the client re-execs
# itself onto the fresh code (closes the stale-.so failure class). Best-effort (--ff-only);
# if the pull/build fails the box stays on old code and is visibly stale in /status.
UPDATE_CMD="cd '$REPO_ROOT' && git pull --ff-only && cd '$ENG' && PATH='$ENG/.venv/bin':\$PATH cpp/build.sh"

# fleet_client owns the engine pool: pull net (weights.bin) + params, spawn + supervise N
# cc_selfplay --jobs-local procs, upload games, contribute gate games, self-update (+ native
# rebuild) on a new server version. worker-id-base 300 -> games attribute to [leena]. The engine
# loads the .bin + reads sims/max-plies/start-fen from the job + env (selfplay.json governs once
# mirrored in), so the arch/device/sims knobs aren't passed here.
CLIENT_ARGS=(
  -m chessckers_engine.fleet_client
  --server "$SERVER" --run-dir "$RUN" --client-id leena --poll-seconds "$FLEET_POLL_S"
  --bind-interface en0
  --update-cmd "$UPDATE_CMD"
  --queue-depth "$FLEET_WORKERS" --spawn-engines
  --engine-binary "$CC_SELFPLAY"
  --engine-workers "$FLEET_WORKERS" --engine-worker-id-base 300 --engine-seed 4000
)

# FOREGROUND=1 -> run the client ATTACHED to this (interactive ssh) session instead of
# detaching it. This is the macOS Local-Network fix: a launchd-orphaned daemon (the default
# nohup/&/disown path below) is DENIED LAN access by macOS privacy, so every connect to the
# LAN server fails EHOSTUNREACH; a process that stays a child of a live, granted ssh session
# reaches the LAN fine. Output streams to your terminal AND run-local/fleet_client.log; Ctrl-C
# (run it under `ssh -t`) stops the client, its workers, and the caffeinate this script started.
if [ -n "${FOREGROUND:-}" ]; then
  rm -f "$RUN/client.pid" 2>/dev/null || true
  cleanup() {
    echo; echo "leena: foreground teardown — stopping engines + caffeinate…"
    touch "$RUN/STOP" 2>/dev/null || true
    pkill -f "cc_selfplay .*--jobs-local" 2>/dev/null || true
    [ -n "${CAFF_PID:-}" ] && kill "$CAFF_PID" 2>/dev/null || true
  }
  trap 'cleanup; exit 0' INT TERM
  echo "leena: FOREGROUND mode -> $SERVER  (Ctrl-C to stop everything). Log also at $RUN/fleet_client.log"
  "$PY" "${CLIENT_ARGS[@]}" 2>&1 | tee "$RUN/fleet_client.log"
  cleanup
  exit 0
fi

# Background (detaches to launchd). NOTE: on a macOS box with Local Network privacy this path
# leaves the client unable to reach a LAN server (orphaned daemons are denied) — use FOREGROUND=1.
nohup "$PY" "${CLIENT_ARGS[@]}" > "$RUN/fleet_client.log" 2>&1 &
echo $! > "$RUN/client.pid"; disown
echo "leena fleet_client launched (pid $(cat "$RUN/client.pid")) -> $SERVER"
echo "leena up: client owns $FLEET_WORKERS cc_selfplay engines (spawned once weights land)."
echo "logs: $RUN/fleet_client.log (client) + $RUN/engine-*.log (engines)"
