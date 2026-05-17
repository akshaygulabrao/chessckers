#!/usr/bin/env bash
# Watchdog for the local training stack.
#
# Polls every $POLL_SEC seconds and restarts any of:
#   - `workers` tmux session (bundled coord: trainer + 1 local worker + eval)
#   - each `sync_*` session listed in scripts/active_remotes.env
#
# Exits cleanly when $LOCAL_RUN/STOP exists (run intentionally ended). When
# the workers coord writes run_summary.json at end-of-run, it also drops a
# STOP file via the signal-handler path or the game-target path, so the
# watchdog will see that and quit instead of relaunching a stopped run.
#
# Launch wrapped in caffeinate so the Mac doesn't idle-sleep:
#   tmux new-session -d -s watchdog "caffeinate -i bash scripts/watchdog.sh"
#
# To add/remove a remote box: edit scripts/active_remotes.env (no code
# change to this file required).
#
# Known limit: this watchdog runs inside tmux; if the tmux server itself
# dies, the watchdog dies with it. For a fully self-healing setup, run
# this via launchd (see scripts/com.chessckers.watchdog.plist).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOCAL_RUN="${LOCAL_RUN:-/Users/ox/AAworkspace/chessckers/engine/runs/local-006}"
POLL_SEC="${POLL_SEC:-60}"
LOG_PATH="${LOG_PATH:-/tmp/watchdog.log}"

# Source the active remotes config. Defines ACTIVE_REMOTES bash array.
# Empty by default (local-only mode); user uncomments lines when a remote
# is bid up.
ACTIVE_REMOTES=()
[ -f scripts/active_remotes.env ] && source scripts/active_remotes.env

log() {
  local line
  line="[$(date +%H:%M:%S)] $*"
  echo "$line"
  echo "$line" >> "$LOG_PATH"
}

restart_workers() {
  log "workers session missing — restarting bundled coord"
  bash scripts/launch_workers.sh local >> "$LOG_PATH" 2>&1
}

restart_remote_sidecar() {
  # Args: "name|host|port|user|remote_run_dir"
  local entry="$1"
  local name host port user run_dir
  IFS='|' read -r name host port user run_dir <<< "$entry"
  log "$name missing — restarting (host=$host:$port user=$user)"
  tmux kill-session -t "$name" 2>/dev/null || true
  tmux new-session -d -s "$name" \
    "CLOUD_HOST='$host' CLOUD_PORT='$port' CLOUD_USER='$user' \
     REMOTE_RUN='$run_dir' LOCAL_RUN='$LOCAL_RUN' SYNC_DOWN=60 \
     bash scripts/cloud_sync_sidecar.sh 2>&1 | tee /tmp/${name}.log"
}

log "watchdog up; LOCAL_RUN=$LOCAL_RUN, POLL_SEC=${POLL_SEC}s, remotes=${#ACTIVE_REMOTES[@]}"

while true; do
  if [ -f "$LOCAL_RUN/STOP" ]; then
    log "STOP file present at $LOCAL_RUN/STOP — exiting watchdog cleanly"
    exit 0
  fi
  tmux has-session -t workers 2>/dev/null || restart_workers
  for entry in "${ACTIVE_REMOTES[@]}"; do
    name="${entry%%|*}"
    tmux has-session -t "$name" 2>/dev/null || restart_remote_sidecar "$entry"
  done
  sleep "$POLL_SEC"
done
