#!/usr/bin/env bash
# Bidirectional sync between local M1 Pro and cloud worker box.
#   - DOWN: rsync /root/run/buffer/ → local-004/buffer/ every $SYNC_DOWN seconds
#   - UP: rsync local-004/weights.pt → /root/run/weights.pt when local mtime changes
#         (workers hot-reload from disk on mtime change)
# Stops on SIGTERM. Use TaskStop on the Monitor.
set -uo pipefail

CLOUD_HOST="${CLOUD_HOST:-ssh2.vast.ai}"
CLOUD_PORT="${CLOUD_PORT:-15564}"
CLOUD_USER="${CLOUD_USER:-root}"
REMOTE_RUN="${REMOTE_RUN:-/root/run}"
LOCAL_RUN="${LOCAL_RUN:-/Users/ox/AAworkspace/chessckers/engine/runs/local-004}"
SYNC_DOWN="${SYNC_DOWN:-60}"

RSYNC_SSH="ssh -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -o ConnectTimeout=10 -p $CLOUD_PORT"
last_w_mtime=0

while true; do
  # 1. Pull new games. --update (mtime-based) instead of --ignore-existing
  # so post-restart games propagate even when their filenames collide with
  # earlier games (workers reset their sequence counter on each relaunch,
  # so worker_id 300 will start writing 300_0000000001.pkl again on restart
  # — different content, same name. --ignore-existing skipped these
  # silently; --update sees the newer mtime and copies through, overwriting
  # the old file in the local buffer).
  before=$(ls "$LOCAL_RUN/buffer" 2>/dev/null | wc -l | tr -d ' ')
  rsync -az --update -e "$RSYNC_SSH" \
    "$CLOUD_USER@$CLOUD_HOST:$REMOTE_RUN/buffer/" "$LOCAL_RUN/buffer/" 2>/dev/null || \
    echo "[sync] rsync down failed at $(date +%H:%M:%S)"
  after=$(ls "$LOCAL_RUN/buffer" 2>/dev/null | wc -l | tr -d ' ')
  delta=$((after - before))

  # 1b. Pull heartbeats. Tiny JSON files (~100 bytes each), one per worker;
  # the coord's authoritative game counter and status dashboard read these.
  mkdir -p "$LOCAL_RUN/heartbeats" 2>/dev/null
  rsync -az --update -e "$RSYNC_SSH" \
    "$CLOUD_USER@$CLOUD_HOST:$REMOTE_RUN/heartbeats/" "$LOCAL_RUN/heartbeats/" 2>/dev/null || true

  # 2. Push weights if changed
  cur_mtime=$(stat -f %m "$LOCAL_RUN/weights.pt" 2>/dev/null || echo 0)
  pushed=""
  if [ "$cur_mtime" -gt "$last_w_mtime" ]; then
    if rsync -az -e "$RSYNC_SSH" \
        "$LOCAL_RUN/weights.pt" "$CLOUD_USER@$CLOUD_HOST:$REMOTE_RUN/weights.pt" 2>/dev/null; then
      last_w_mtime=$cur_mtime
      pushed="  weights→remote"
    fi
  fi

  if [ "$delta" -gt 0 ] || [ -n "$pushed" ]; then
    echo "[$(date +%H:%M:%S)] +${delta} games (total=$after)$pushed"
  fi

  sleep $SYNC_DOWN
done
