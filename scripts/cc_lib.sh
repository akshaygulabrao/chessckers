#!/usr/bin/env bash
# Shared helpers for the local launch/stop scripts (launch_next.sh, stop_run.sh).
# Source it after ENG is set; it falls back to <repo>/engine otherwise.

: "${ENG:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../engine" && pwd)}"
LOG=/tmp/cc_train.log          # SINGLE unified run log (trainer + workers + leena_sync)
COUNTER=/tmp/cc_gamecount      # shared game counter (mirrors selfplay_az_loop._GAME_COUNTER)
VENV="$ENG/.venv/bin/python"

# Kill orphaned (ppid==1) chessckers mp-spawn worker children that a non-graceful
# stop (pkill -9 on the parent) can leave behind — their cmdline carries
# "multiprocessing.spawn", not "selfplay_workers_only", so a name-based pkill
# misses them. Precise: only this venv's procs. Prints a line iff it swept any.
sweep_orphans() {
  local p ppid cmd swept=0
  for p in $(pgrep -f 'multiprocessing.(spawn|resource_tracker)' 2>/dev/null); do
    read -r ppid cmd < <(ps -o ppid=,command= -p "$p" 2>/dev/null)
    [ "$ppid" = "1" ] && case "$cmd" in
      *"$VENV"*) kill -9 "$p" 2>/dev/null && swept=$((swept + 1)) ;;
    esac
  done
  [ "$swept" -gt 0 ] && echo "[sweep] killed $swept orphaned worker(s)"
  return 0
}
