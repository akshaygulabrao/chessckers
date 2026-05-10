#!/usr/bin/env bash
# Unified self-play worker launcher. One script, three targets.
#
# Usage:
#   scripts/launch_workers.sh local
#   scripts/launch_workers.sh leena
#   scripts/launch_workers.sh vast <ssh_host> <ssh_port>
#
# Pipeline (skips ssh/git push when HOST=local):
#   1. Refuse to deploy a dirty working tree.
#   2. git push ssh://user@host:port/<bare> HEAD:deploy
#      (atomic; reuses your ssh key — no GitHub or third-party auth.)
#   3. Remote: git fetch + reset --hard FETCH_HEAD into the working clone.
#   4. uv pip install --python $VENV_DIR/bin/python -e ./engine
#      (respects pre-installed torch on vast's pytorch image; installs
#       everything on Leena's bare-Mac venv.)
#   5. Launch workers (and optional pruner) in detached tmux sessions.
set -euo pipefail

TARGET="${1:?Usage: $0 <local|leena|vast> [host] [port]}"
shift

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$REPO_ROOT/scripts/targets/$TARGET.env"
[ -f "$ENV_FILE" ] || { echo "Missing env file: $ENV_FILE" >&2; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"

# CLI args override env (vast has variable host/port per instance).
HOST="${1:-${HOST:-}}"
PORT="${2:-${PORT:-22}}"
[ -n "$HOST" ] || { echo "HOST required (set in env file or pass as arg)" >&2; exit 1; }

# PYTHON_BIN may be set directly by the env file (vast: /opt/conda/bin/python3,
# pre-baked torch). Otherwise we derive it from VENV_DIR (local/leena: managed
# by uv).
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

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

# ---- 1. source sync (git push over ssh) ----------------------------------
if [ "$HOST" != "local" ]; then
  if [ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]; then
    log "ABORT: working tree has uncommitted changes. Commit before deploying."
    git -C "$REPO_ROOT" status --short >&2
    exit 1
  fi
  log "[1/3] git push -> $USER@$HOST:$REMOTE_BARE"
  remote "test -d '$REMOTE_BARE' || git init --bare '$REMOTE_BARE'"
  git -C "$REPO_ROOT" push --force \
    "ssh://$USER@$HOST:$PORT$REMOTE_BARE" HEAD:deploy
  remote "test -d '$WORK_DIR/.git' || git clone '$REMOTE_BARE' '$WORK_DIR'"
  remote "cd '$WORK_DIR' && git fetch origin deploy && git reset --hard FETCH_HEAD"
fi

# ---- 2. deps -------------------------------------------------------------
# USE_PIP=1 (vast): plain `pip install -e .` against the pre-baked conda
# Python. pip sees existing torch/httpx/chess/numpy and skips them; just
# registers the chessckers_engine editable. No uv needed.
# USE_PIP=0 (local/leena): uv manages a project venv and installs deps.
if [ "${USE_PIP:-0}" = "1" ]; then
  log "[2/3] pip install -e engine (skips deps already satisfied)"
  remote "'$PYTHON_BIN' -m pip install --quiet -e '$WORK_DIR/engine'"
else
  log "[2/3] uv pip install --python $PYTHON_BIN -e engine"
  remote "command -v uv >/dev/null || { echo 'uv not installed on target' >&2; exit 1; }; \
    test -x '$PYTHON_BIN' || uv venv '$VENV_DIR' --python 3.11; \
    uv pip install --python '$PYTHON_BIN' -e '$WORK_DIR/engine'"
fi

# ---- 3. launch in tmux ---------------------------------------------------
# Args common to both modes (model arch, MCTS hyper-params, shared inference).
SHARED_ARGS="--run-dir '$RUN_DIR' \
  --workers $N_WORKERS \
  --device $DEVICE --sims $SIMS \
  $SHARED_INFERENCE_FLAGS \
  --d-hidden 256 --c-filters 128 --n-blocks 6 \
  --temperature 1.0 --dirichlet-alpha 0.5 --dirichlet-eps 0.40 \
  --mcts-batch-size 8 --vloss-batch 8 --max-plies 400 \
  --seed $SEED"

case "${MODE:-workers_only}" in
  bundled)
    # selfplay_az_async = workers + trainer + eval in one supervised process.
    MODULE="chessckers_engine.selfplay_az_async"
    MODE_ARGS="--run-seconds ${RUN_SECONDS:-86400}"
    [ -n "${RESUME_FROM:-}" ] && MODE_ARGS="$MODE_ARGS --resume-from '$RESUME_FROM'"
    [ -n "${BASE_WEIGHTS:-}" ] && MODE_ARGS="$MODE_ARGS --base '$BASE_WEIGHTS'"
    ;;
  workers_only)
    # Pure inference farm. Reads $RUN_DIR/weights.pt, polls for updates.
    MODULE="chessckers_engine.selfplay_workers_only"
    MODE_ARGS="--weights '$RUN_DIR/weights.pt' \
      --worker-id-base $WID_BASE \
      --weights-poll-seconds 30"
    ;;
  *)
    echo "Unknown MODE=$MODE (expected: bundled | workers_only)" >&2
    exit 1
    ;;
esac

log "[3/3] launch $MODULE (MODE=$MODE) in tmux session 'workers'"
CMD="cd '$WORK_DIR/engine' && exec '$PYTHON_BIN' -m $MODULE $SHARED_ARGS $MODE_ARGS"

remote "mkdir -p '$RUN_DIR'; tmux kill-session -t workers 2>/dev/null; \
  tmux new-session -d -s workers \"$CMD\""

if [ -n "${PRUNER_KEEP:-}" ]; then
  log "  + pruner (keep newest $PRUNER_KEEP in $RUN_DIR/buffer)"
  PRUNE_CMD="cd '$RUN_DIR/buffer' && while true; do \
    ls -t *.pkl 2>/dev/null | tail -n +$((PRUNER_KEEP+1)) | xargs -r rm -f; \
    sleep 60; \
  done"
  remote "mkdir -p '$RUN_DIR/buffer'; tmux kill-session -t pruner 2>/dev/null; \
    tmux new-session -d -s pruner \"$PRUNE_CMD\""
fi

sleep 2
if remote "tmux has-session -t workers 2>/dev/null"; then
  log "OK: workers up on $TARGET ($HOST)"
  if [ "$HOST" = "local" ]; then
    log "  logs:  tmux capture-pane -t workers -p | tail -30"
    log "  stop:  tmux kill-session -t workers"
  else
    log "  logs:  ssh -p $PORT $USER@$HOST 'tmux capture-pane -t workers -p | tail -30'"
    log "  stop:  ssh -p $PORT $USER@$HOST 'tmux kill-session -t workers'"
  fi
else
  log "FAIL: tmux session 'workers' didn't survive 2s — capturing pane:"
  remote "tmux capture-pane -t workers -p 2>/dev/null || true" >&2
  exit 1
fi
