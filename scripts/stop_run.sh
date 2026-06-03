#!/usr/bin/env bash
# Gracefully stop the local launch_next.sh run (trainer + workers + leena_sync)
# WITHOUT orphaning worker children.
#
# The orphan leak: `pkill -9 -f selfplay_workers_only` kills the worker PARENT,
# but its multiprocessing.spawn CHILDREN don't carry that string in their
# cmdline — they survive, reparent to init, and keep pegging the CPU.
#
# Fix: signal via the STOP file. selfplay_workers_only + train_continuous both
# watch <run-dir>/STOP and exit cleanly, so the parent reaps its children (no
# orphans). A final sweep kills any orphaned mp-spawn children as a safety net.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENG="$REPO_ROOT/engine"
RUN_DIR="${RUN_DIR:-$ENG/weights/run}"
VENV="$ENG/.venv/bin/python"
LEENA="${LEENA:-leenagulabrao@Leenas-MacBook-Air.local}"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"

echo "[stop] STOP -> $RUN_DIR/STOP (graceful: workers + trainer finish + exit)"
touch "$RUN_DIR/STOP"

# leena_sync watches STOP and tears down leena's workers, then exits. Workers
# only check STOP between games, so an in-flight 200-ply game can delay exit by
# a minute+ — wait generously (the worker parent's own deadline is 300s).
for _ in $(seq 240); do
  pgrep -f 'chessckers_engine.(train_continuous|selfplay_workers_only)' >/dev/null || break
  sleep 1
done
pkill -f "$ENG/scripts/leena_sync.sh" 2>/dev/null || true

# Safety net: sweep ORPHANED (ppid=1) chessckers mp-spawn children the graceful
# exit may have missed. Precise: only this venv's mp-spawn/resource_tracker procs.
swept=0
for p in $(pgrep -f 'multiprocessing.(spawn|resource_tracker)' 2>/dev/null); do
  cmd=$(ps -o command= -p "$p" 2>/dev/null)
  ppid=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  case "$cmd" in
    *"$VENV"*) [ "$ppid" = "1" ] && { kill -9 "$p" 2>/dev/null && swept=$((swept+1)); } ;;
  esac
done
[ "$swept" -gt 0 ] && echo "[stop] swept $swept orphaned worker(s)"

# Belt-and-suspenders for leena (leena_sync already does this on STOP).
$SSH "$LEENA" 'pkill -f selfplay_workers_only; pkill -f multiprocessing.spawn' 2>/dev/null || true

rm -f "$RUN_DIR/STOP"
echo "[stop] done. remaining:"
pgrep -fl 'chessckers_engine.(train_continuous|selfplay_workers_only)' || echo "  (none)"
