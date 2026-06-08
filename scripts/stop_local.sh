#!/usr/bin/env bash
# Gracefully stop the local fleet (trainer + arena + fleet_server + the loopback self-play
# client) via the STOP file: touching run/STOP makes the server advertise STOP, so every
# client (local + leena) winds its workers down and exits — no orphaned mp-spawn children —
# then we sweep any stragglers. NOTE: this stops the WHOLE fleet, leena included (STOP is
# the fleet-wide control signal relayed through the server).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENG="$REPO_ROOT/engine"
RUN="${RUN:-$ENG/weights/run}"            # server-side run-dir (trainer / arena / server)
CRUN="${CRUN:-$ENG/weights/run-local}"    # local client's own run-dir
say(){ echo "[stop-local] $*" >&2; }

say "STOP -> $RUN/STOP (graceful: trainer + arena + server + clients finish + exit)"
touch "$RUN/STOP"

# Clients see STOP via the server within a poll (~15s); the cc_selfplay engines only check
# between games, so a 200-ply game can delay exit ~a minute. Wait for all of them to drain.
for _ in $(seq 240); do
  pgrep -f 'chessckers_engine\.(train_continuous|fleet_arena|fleet_client)' >/dev/null || break
  sleep 1
done

# Safety net: kill any straggler parents + orphaned cc_selfplay engines + mp-spawn children.
pkill -f 'chessckers_engine\.(train_continuous|fleet_arena|fleet_server|fleet_client)' 2>/dev/null || true
pkill -f 'cc_selfplay .*--jobs-local' 2>/dev/null || true
pkill -f 'multiprocessing.spawn' 2>/dev/null || true

rm -f "$RUN/STOP" "$CRUN/STOP" 2>/dev/null || true
say "done. remaining:"
pgrep -fl 'chessckers_engine\.(train_continuous|fleet_arena|fleet_server|fleet_client)' || say "  (none)"
