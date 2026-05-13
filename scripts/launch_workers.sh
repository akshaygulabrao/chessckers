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
    # Prepend /opt/homebrew/bin + /usr/local/bin so brew-installed tools
    # (tmux, uv) on macOS remotes are findable — zsh over non-interactive
    # ssh only sources .zshenv, not .zprofile where brew shellenv lives.
    # On Linux remotes those paths don't exist; harmless prepend.
    ssh "${SSH_OPTS[@]}" -p "$PORT" "$USER@$HOST" \
      "export PATH=/opt/homebrew/bin:/usr/local/bin:\$PATH; $1"
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

# ---- 2b. N_WORKERS=auto: derive from cgroup / sysctl --------------------
# On vast, nproc returns the HOST core count (e.g. 64), not the bid's
# effective allotment — cgroup CPU quota is the real cap. On macOS, the
# cgroup files don't exist; sysctl is truth. Subtract RESERVE_CORES so
# the inference server thread + pruner + ssh have headroom.
if [ "${N_WORKERS:-}" = "auto" ]; then
  CORES=$(remote 'if [ -f /sys/fs/cgroup/cpu.max ]; then
    read -r q p < /sys/fs/cgroup/cpu.max
    [ "$q" != "max" ] && [ -n "$p" ] && echo $((q/p)) && exit 0
  fi
  if [ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    [ "$q" != "-1" ] && [ -n "$p" ] && echo $((q/p)) && exit 0
  fi
  command -v sysctl >/dev/null && sysctl -n hw.ncpu 2>/dev/null && exit 0
  nproc' | tr -d '[:space:]')
  RESERVE="${RESERVE_CORES:-2}"
  N_WORKERS=$(( CORES - RESERVE ))
  [ "$N_WORKERS" -lt 1 ] && N_WORKERS=1
  log "N_WORKERS=auto -> $N_WORKERS  (effective cores=$CORES, reserve=$RESERVE)"
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
    # --eval-workers 1: keep the periodic NN-vs-random eval cheap (default
    # was 4, which spawned 4 extra worker processes for ~minutes during
    # each eval cycle — noticeable on a quiet dev laptop).
    MODULE="chessckers_engine.selfplay_az_async"
    MODE_ARGS="--run-seconds ${RUN_SECONDS:-86400} --eval-workers ${EVAL_WORKERS:-1}"
    # Prefer the latest durable checkpoint written by THIS run over RESUME_FROM.
    # RESUME_FROM is the *seed* for the first launch in a fresh run dir; once
    # the trainer has written checkpoints into $RUN_DIR/checkpoints/, every
    # subsequent restart should pick those up so we don't redo the steps
    # between RESUME_FROM and the last checkpoint.
    #
    # We sort by filename (lexical, descending), NOT mtime. Filenames are
    # 'step_NNNNNNNN[_final].pt' with zero-padded step numbers, so lexical
    # sort matches step order. mtime is unreliable here: a restart that
    # re-passes a step (e.g. trainer resumed from N-2k and re-saves step N)
    # would re-touch an older checkpoint file and look "latest" by mtime,
    # even though a higher-step checkpoint from the previous incarnation
    # still exists with an older mtime.
    LATEST_CKPT=$(remote "ls '$RUN_DIR/checkpoints/'*.pt 2>/dev/null | sort -r | head -1" | tr -d '[:space:]')
    if [ -n "$LATEST_CKPT" ]; then
      log "  resume: $LATEST_CKPT (latest local; overrides RESUME_FROM=$RESUME_FROM)"
      MODE_ARGS="$MODE_ARGS --resume-from '$LATEST_CKPT'"
    elif [ -n "${RESUME_FROM:-}" ]; then
      log "  resume: $RESUME_FROM (no local checkpoints yet)"
      MODE_ARGS="$MODE_ARGS --resume-from '$RESUME_FROM'"
    fi
    [ -n "${BASE_WEIGHTS:-}" ] && MODE_ARGS="$MODE_ARGS --base '$BASE_WEIGHTS'"
    [ -n "${RUN_GAMES:-}" ] && MODE_ARGS="$MODE_ARGS --run-games $RUN_GAMES"
    [ -n "${EXTRA_BUNDLED_ARGS:-}" ] && MODE_ARGS="$MODE_ARGS $EXTRA_BUNDLED_ARGS"
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
# OMP/MKL=1 prevents PyTorch oversubscription: each of N worker processes
# would otherwise default to ~N OMP threads, totalling N^2 threads
# fighting for the same cores. Single-thread per worker is correct when
# we have many worker processes (i.e. per-worker inference mode); harmless
# when --shared-inference is set (only the server thread does forwards).
CMD="cd '$WORK_DIR/engine' && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  exec '$PYTHON_BIN' -m $MODULE $SHARED_ARGS $MODE_ARGS"

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
