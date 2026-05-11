#!/usr/bin/env bash
# Bring up the full distributed training system in one shot:
#   - local (bundled): trainer + workers + eval in selfplay_az_async
#   - vast  (workers_only): pure inference farm
#   - sync sidecar (local tmux 'sync_vast'): pushes weights up, pulls buffer down
#
# Idempotent: aborts if any of the three pieces is already running.
#
# Usage:
#   scripts/launch_all.sh <vast_instance_id>
#
# Resolves direct SSH (public_ipaddr + direct_port_end) from the instance
# ID via `vastai ssh-url` — never uses the proxy ssh*.vast.ai endpoint
# (proxy is unreliable; see feedback memory).
#
# Env:
#   SKIP_SYNC=1   don't start the sync sidecar (you'll need it eventually
#                 or cloud workers stay idle waiting for weights.pt).
set -euo pipefail

VAST_INSTANCE="${1:?Usage: $0 <vast_instance_id>}"

# Resolve direct SSH. `vastai ssh-url` returns ssh://root@<ip>:<port>.
SSH_URL=$(vastai ssh-url "$VAST_INSTANCE" 2>&1)
case "$SSH_URL" in
  ssh://*) ;;
  *) echo "ABORT: could not resolve ssh-url for instance $VAST_INSTANCE: $SSH_URL" >&2; exit 1 ;;
esac
# Strip ssh:// then split user@host:port.
SSH_REST="${SSH_URL#ssh://*@}"
VAST_HOST="${SSH_REST%:*}"
VAST_PORT="${SSH_REST##*:}"
echo "[$(date +%H:%M:%S)] resolved instance $VAST_INSTANCE -> $VAST_HOST:$VAST_PORT (direct)" >&2

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o LogLevel=ERROR
          -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=3)

# session_exists <session> <host> [port] [user]
# Returns 0 if a tmux session by that name exists on the target.
session_exists() {
  local session="$1" host="$2" port="${3:-}" user="${4:-}"
  if [ "$host" = "local" ]; then
    tmux has-session -t "$session" 2>/dev/null
  else
    ssh "${SSH_OPTS[@]}" -p "$port" "$user@$host" \
      "tmux has-session -t $session 2>/dev/null"
  fi
}

# ---- 1. Idempotency checks. --------------------------------------------
log "[1/4] verify nothing is already running"

if session_exists workers local; then
  log "ABORT: local already has a 'workers' tmux session."
  log "       stop it first: scripts/stop_workers.sh local"
  exit 1
fi

# Fail fast on ssh connectivity before reporting "vast not running".
ssh "${SSH_OPTS[@]}" -p "$VAST_PORT" "root@$VAST_HOST" true \
  || { log "ABORT: cannot ssh root@$VAST_HOST:$VAST_PORT"; exit 1; }

if session_exists workers "$VAST_HOST" "$VAST_PORT" root; then
  log "ABORT: vast already has a 'workers' tmux session."
  log "       stop it first: scripts/stop_workers.sh vast $VAST_HOST $VAST_PORT"
  exit 1
fi

if [ -z "${SKIP_SYNC:-}" ] && session_exists sync_vast local; then
  log "ABORT: local already has a 'sync_vast' tmux session."
  log "       stop it first: tmux kill-session -t sync_vast"
  exit 1
fi

# ---- 2. Launch local (bundled). ----------------------------------------
log "[2/4] launching local (bundled: trainer + workers)"
"$REPO_ROOT/scripts/launch_workers.sh" local

# ---- 3. Launch vast (workers_only). ------------------------------------
log "[3/4] launching vast (workers_only)"
"$REPO_ROOT/scripts/launch_workers.sh" vast "$VAST_HOST" "$VAST_PORT"

# ---- 4. Launch sync sidecar (weights up / buffer down). ----------------
if [ -z "${SKIP_SYNC:-}" ]; then
  log "[4/4] launching sync sidecar (local tmux 'sync_vast')"
  # Capture LOCAL_RUN before vast.env sources clobber RUN_DIR.
  # shellcheck disable=SC1091
  source "$REPO_ROOT/scripts/targets/local.env"
  LOCAL_RUN_DIR="$RUN_DIR"
  # shellcheck disable=SC1091
  source "$REPO_ROOT/scripts/targets/vast.env"
  REMOTE_RUN_DIR="$RUN_DIR"

  SYNC_CMD="CLOUD_HOST='$VAST_HOST' CLOUD_PORT='$VAST_PORT' CLOUD_USER='root' \
    REMOTE_RUN='$REMOTE_RUN_DIR' LOCAL_RUN='$LOCAL_RUN_DIR' \
    '$REPO_ROOT/scripts/cloud_sync_sidecar.sh'"

  tmux new-session -d -s sync_vast "$SYNC_CMD"
  sleep 1
  if tmux has-session -t sync_vast 2>/dev/null; then
    log "  sync sidecar up"
  else
    log "  WARN: sync sidecar didn't survive — check by hand"
  fi
fi

log "DONE."
log "  local logs:  tmux capture-pane -t workers -p | tail -40"
log "  vast  logs:  ssh -p $VAST_PORT root@$VAST_HOST 'tmux capture-pane -t workers -p | tail -40'"
log "  sync  logs:  tmux capture-pane -t sync_vast -p | tail -20"
log "  stop all:    scripts/stop_workers.sh local && \\"
log "               scripts/stop_workers.sh vast $VAST_HOST $VAST_PORT && \\"
log "               tmux kill-session -t sync_vast"
