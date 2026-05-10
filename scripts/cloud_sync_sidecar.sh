#!/usr/bin/env bash
# Bidirectional sync between local M1 Pro and cloud worker box.
#   - DOWN: rsync /root/run/buffer/ → local-004/buffer/ every $SYNC_DOWN seconds
#   - UP: rsync local-004/weights.pt → /root/run/weights.pt when local mtime changes
#         (workers hot-reload from disk on mtime change)
# Stops on SIGTERM. Use TaskStop on the Monitor.
set -uo pipefail

CLOUD_HOST="${CLOUD_HOST:-ssh2.vast.ai}"
CLOUD_PORT="${CLOUD_PORT:-15564}"
LOCAL_RUN="${LOCAL_RUN:-/Users/ox/AAworkspace/chessckers/engine/runs/local-004}"
SYNC_DOWN="${SYNC_DOWN:-60}"

RSYNC_SSH="ssh -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -o ConnectTimeout=10 -p $CLOUD_PORT"
last_w_mtime=0

while true; do
  # 1. Pull new games (incremental rsync)
  before=$(ls "$LOCAL_RUN/buffer" 2>/dev/null | wc -l | tr -d ' ')
  rsync -az --ignore-existing -e "$RSYNC_SSH" \
    "root@$CLOUD_HOST:/root/run/buffer/" "$LOCAL_RUN/buffer/" 2>/dev/null || \
    echo "[sync] rsync down failed at $(date +%H:%M:%S)"
  after=$(ls "$LOCAL_RUN/buffer" 2>/dev/null | wc -l | tr -d ' ')
  delta=$((after - before))

  # 2. Push weights if changed
  cur_mtime=$(stat -f %m "$LOCAL_RUN/weights.pt" 2>/dev/null || echo 0)
  pushed=""
  if [ "$cur_mtime" -gt "$last_w_mtime" ]; then
    if rsync -az -e "$RSYNC_SSH" \
        "$LOCAL_RUN/weights.pt" "root@$CLOUD_HOST:/root/run/weights.pt" 2>/dev/null; then
      last_w_mtime=$cur_mtime
      pushed="  weights→cloud"
    fi
  fi

  if [ "$delta" -gt 0 ] || [ -n "$pushed" ]; then
    echo "[$(date +%H:%M:%S)] +${delta} games (total=$after)$pushed"
  fi

  sleep $SYNC_DOWN
done
