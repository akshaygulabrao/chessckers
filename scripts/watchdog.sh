#!/usr/bin/env bash
# Watchdog for the local-006 training stack.
#
# Polls every $POLL_SEC seconds and restarts any of:
#   - `workers` tmux session (bundled coord: trainer + 1 local worker + eval)
#   - `sync_leena` tmux session (bidirectional rsync to Leena LAN box)
#   - `sync_vast`  tmux session (bidirectional rsync to vast.ai box)
#
# Exits cleanly when $LOCAL_RUN/STOP exists (run intentionally ended). When
# the workers coord writes run_summary.json at end-of-run, it also drops a
# STOP file via the signal-handler path or the game-target path, so the
# watchdog will see that and quit instead of relaunching a stopped run.
#
# Launch wrapped in caffeinate so the Mac doesn't idle-sleep:
#   tmux new-session -d -s watchdog "caffeinate -i bash scripts/watchdog.sh"
#
# Known limit: this watchdog runs inside tmux; if the tmux server itself
# dies, the watchdog dies with it. For a fully self-healing setup, run
# this via launchd instead. Good enough for an overnight run.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOCAL_RUN="${LOCAL_RUN:-/Users/ox/AAworkspace/chessckers/engine/runs/local-006}"
POLL_SEC="${POLL_SEC:-60}"

# Per-target settings inlined here. If we end up adding more sidecars
# (e.g. a second vast box), refactor to an array; for three sessions it's
# clearer to read the explicit functions.
LEENA_HOST="${LEENA_HOST:-192.168.68.183}"
LEENA_PORT="${LEENA_PORT:-22}"
LEENA_USER="${LEENA_USER:-leenagulabrao}"
LEENA_RUN="${LEENA_RUN:-/Users/leenagulabrao/chessckers/engine/run}"

VAST_HOST="${VAST_HOST:-220.82.52.202}"
VAST_PORT="${VAST_PORT:-52232}"
VAST_USER="${VAST_USER:-root}"
VAST_RUN="${VAST_RUN:-/root/run}"

LOG_PATH="${LOG_PATH:-/tmp/watchdog.log}"

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

restart_sync_leena() {
  log "sync_leena missing — restarting"
  tmux kill-session -t sync_leena 2>/dev/null || true
  tmux new-session -d -s sync_leena \
    "CLOUD_HOST='$LEENA_HOST' CLOUD_PORT='$LEENA_PORT' CLOUD_USER='$LEENA_USER' \
     REMOTE_RUN='$LEENA_RUN' LOCAL_RUN='$LOCAL_RUN' SYNC_DOWN=60 \
     bash scripts/cloud_sync_sidecar.sh 2>&1 | tee /tmp/sync_leena.log"
}

restart_sync_vast() {
  log "sync_vast missing — restarting"
  tmux kill-session -t sync_vast 2>/dev/null || true
  tmux new-session -d -s sync_vast \
    "CLOUD_HOST='$VAST_HOST' CLOUD_PORT='$VAST_PORT' CLOUD_USER='$VAST_USER' \
     REMOTE_RUN='$VAST_RUN' LOCAL_RUN='$LOCAL_RUN' SYNC_DOWN=60 \
     bash scripts/cloud_sync_sidecar.sh 2>&1 | tee /tmp/sync_vast.log"
}

log "watchdog up; LOCAL_RUN=$LOCAL_RUN, POLL_SEC=${POLL_SEC}s"

while true; do
  if [ -f "$LOCAL_RUN/STOP" ]; then
    log "STOP file present at $LOCAL_RUN/STOP — exiting watchdog cleanly"
    exit 0
  fi
  tmux has-session -t workers    2>/dev/null || restart_workers
  tmux has-session -t sync_leena 2>/dev/null || restart_sync_leena
  tmux has-session -t sync_vast  2>/dev/null || restart_sync_vast
  sleep "$POLL_SEC"
done
