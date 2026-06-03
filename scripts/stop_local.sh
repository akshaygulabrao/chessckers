#!/usr/bin/env bash
# Gracefully stop the local keep-best fleet (trainer + arena + workers) via the
# STOP file so mp-spawn worker children are reaped (no orphans), then sweep any
# stragglers. Mirrors stop_run.sh but for the no-HTTP local topology.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENG="$REPO_ROOT/engine"
RUN="${RUN:-$ENG/weights/run}"
say(){ echo "[stop-local] $*" >&2; }

say "STOP -> $RUN/STOP (graceful: trainer + arena + workers finish their game/step and exit)"
touch "$RUN/STOP"

# Workers only check STOP between games; a 200-ply game can delay exit ~a minute.
for _ in $(seq 240); do
  pgrep -f 'chessckers_engine\.(train_continuous|fleet_arena|selfplay_workers_only)' >/dev/null || break
  sleep 1
done

# Safety net: kill any straggler parents + orphaned mp-spawn children.
pkill -f 'chessckers_engine\.(train_continuous|fleet_arena|fleet_server|fleet_client|selfplay_workers_only)' 2>/dev/null || true
pkill -f 'multiprocessing.spawn' 2>/dev/null || true

rm -f "$RUN/STOP"
say "done. remaining:"
pgrep -fl 'chessckers_engine\.(train_continuous|fleet_arena|selfplay_workers_only)' || say "  (none)"
