#!/usr/bin/env bash
# Graceful counterpart to launch_workers.sh.
#
# Usage:
#   scripts/stop_workers.sh local
#   scripts/stop_workers.sh leena
#   scripts/stop_workers.sh vast <ssh_host> <ssh_port>
#
# Sequence:
#   1. touch $RUN_DIR/STOP — workers finish in-flight games then exit
#      (no half-written .pkl files in buffer/).
#   2. Wait for $RUN_DIR/exit_code to appear (up to STOP_TIMEOUT seconds).
#   3. tmux kill-session for workers + pruner (the panes have already exited;
#      this just cleans up the empty sessions).
set -euo pipefail

TARGET="${1:?Usage: $0 <local|leena|vast> [host] [port]}"
shift

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$REPO_ROOT/scripts/targets/$TARGET.env"
[ -f "$ENV_FILE" ] || { echo "Missing env file: $ENV_FILE" >&2; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"

HOST="${1:-${HOST:-}}"
PORT="${2:-${PORT:-22}}"
STOP_TIMEOUT="${STOP_TIMEOUT:-300}"

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o LogLevel=ERROR
          -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=3)
[ -n "${SSH_KEY:-}" ] && SSH_OPTS+=(-i "$SSH_KEY")

remote() {
  if [ "$HOST" = "local" ]; then
    bash -lc "$1"
  else
    ssh "${SSH_OPTS[@]}" -p "$PORT" "$USER@$HOST" "$1"
  fi
}

log "[1/3] touch $RUN_DIR/STOP — workers will finish in-flight games"
remote "rm -f '$RUN_DIR/exit_code'; touch '$RUN_DIR/STOP'"

log "[2/3] waiting for exit_code (timeout ${STOP_TIMEOUT}s)"
remote "for i in \$(seq 1 $STOP_TIMEOUT); do \
  [ -f '$RUN_DIR/exit_code' ] && { echo \"workers exited cleanly (code=\$(cat '$RUN_DIR/exit_code'))\"; exit 0; }; \
  sleep 1; \
done; \
echo 'TIMEOUT: workers did not write exit_code in time'; exit 1"

log "[3/3] cleaning up tmux sessions"
remote "tmux kill-session -t workers 2>/dev/null; tmux kill-session -t pruner 2>/dev/null; true"
remote "rm -f '$RUN_DIR/STOP'"

log "OK: stopped on $TARGET ($HOST)"
