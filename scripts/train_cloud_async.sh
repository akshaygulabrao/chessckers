#!/usr/bin/env bash
# scripts/train_cloud_async.sh
#
# Vast.ai driver for the **async** AlphaZero coordinator.
# Workers and trainer run continuously, gating dropped, 800 sims, 5M-param net.
# Mirrors the STEP gating of train_cloud.sh so phases are independently
# rerunnable.
#
# Usage:
#   SSH_STRING="ssh -p 22080 root@1.2.3.4" STEP=1 ./scripts/train_cloud_async.sh
#   SSH_STRING="ssh -p 22080 root@1.2.3.4" STEP=all ./scripts/train_cloud_async.sh
#   STEP=4-status VAST_HOST=... VAST_PORT=... ./scripts/train_cloud_async.sh

set -euo pipefail

STEP="${STEP:-help}"

if [[ "$STEP" != "help" && "$STEP" != "" ]]; then
  if [[ -z "${VAST_HOST:-}" || -z "${VAST_PORT:-}" ]]; then
    if [[ -n "${SSH_STRING:-}" ]]; then
      VAST_PORT="$(printf '%s\n' "$SSH_STRING" | grep -oE -- '-p[[:space:]]*[0-9]+' | grep -oE '[0-9]+')"
      VAST_HOST="$(printf '%s\n' "$SSH_STRING" | grep -oE 'root@[^ ]+' | sed 's/root@//')"
    else
      echo "ERROR: set VAST_HOST and VAST_PORT, or pass SSH_STRING='ssh -p PORT root@HOST'" >&2
      exit 2
    fi
  fi
fi
VAST_USER="${VAST_USER:-root}"
VAST_HOST="${VAST_HOST:-}"
VAST_PORT="${VAST_PORT:-}"
export VAST_HOST VAST_PORT VAST_USER

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o LogLevel=ERROR)
if [[ -n "$VAST_PORT" ]]; then
  SSH=(ssh "${SSH_OPTS[@]}" -p "$VAST_PORT" "$VAST_USER@$VAST_HOST")
  RSYNC_SSH="ssh ${SSH_OPTS[*]} -p $VAST_PORT"
fi

# === paths ===
LOCAL_REPO="/Users/ox/AAworkspace/chessckers"
LOCAL_ENGINE="$LOCAL_REPO/engine"
REMOTE_REPO="/root/chessckers"
REMOTE_ENGINE="$REMOTE_REPO/engine"
META_DIR="/root/run-meta"

# === pinned framework versions (match local venv) ===
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
CHESS_VERSION="${CHESS_VERSION:-1.11.2}"

# === run config (env overrides) ===
RUN_NAME="${RUN_NAME:-async-001}"
# Async coordinator hyperparameters — defaults from the M5 plan.
WORKERS="${WORKERS:-8}"
SIMS="${SIMS:-800}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.5}"
DIRICHLET_EPS="${DIRICHLET_EPS:-0.40}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_PLIES="${MAX_PLIES:-400}"
MCTS_BATCH_SIZE="${MCTS_BATCH_SIZE:-8}"
VLOSS_BATCH="${VLOSS_BATCH:-8}"
# 5M-param network (down from the 30M of cloud-run-001 — small-batch GPU
# inference is memory-bandwidth bound, smaller net = more games/dollar).
MODEL_BLOCKS="${MODEL_BLOCKS:-6}"
MODEL_FILTERS="${MODEL_FILTERS:-128}"
MODEL_HIDDEN="${MODEL_HIDDEN:-256}"
TRAIN_BATCH="${TRAIN_BATCH:-128}"
TRAIN_LR="${TRAIN_LR:-1e-3}"
WEIGHT_SAVE_EVERY="${WEIGHT_SAVE_EVERY:-200}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2000}"
MIN_BUFFER_GAMES="${MIN_BUFFER_GAMES:-20}"
BUFFER_MAX_GAMES="${BUFFER_MAX_GAMES:-4000}"
# Eval defaults are deliberately cheap. With workers + trainer all sharing
# one GPU, an "ideal" 20-games × 200-sims eval block grew to >30 min in
# practice — and since eval interval was 30 min, evals overlapped and
# permanently choked self-play. 4 games × 50 sims × 2 eval-workers = ~1 min.
EVAL_EVERY_SECONDS="${EVAL_EVERY_SECONDS:-3600}"
EVAL_GAMES="${EVAL_GAMES:-4}"
EVAL_SIMS="${EVAL_SIMS:-50}"
EVAL_WORKERS="${EVAL_WORKERS:-2}"
RUN_SECONDS="${RUN_SECONDS:-86400}"   # 24h
SEED="${SEED:-0}"

case "$STEP" in
  help|"")
    cat <<EOF
Usage: STEP=<step> SSH_STRING='ssh -p PORT root@HOST' $0

Steps:
  1          Sanity (GPU, python, disk, torch)
  2          Install torch+chess pinned to local, rsync engine code, pip install -e
  4          Launch async coordinator detached
  4-status   Token-cheap status: pid + alive + last few log lines
  4-wait     Block ON THE REMOTE until exit_code lands (run with run_in_background)
  5          Rsync runs/<RUN_NAME>/ + train.log back
  5-up       Rsync local runs/<RUN_NAME>/ UP to remote (resume after preemption)
  6          Destroy vast.ai instance (auto-detects ID from VAST_HOST)
  all        1 → 2 → 4 → 4-wait → 5 → 6 (destroy unconditional)

Tunables:
  RUN_NAME=$RUN_NAME WORKERS=$WORKERS SIMS=$SIMS RUN_SECONDS=$RUN_SECONDS
  DIRICHLET_ALPHA=$DIRICHLET_ALPHA DIRICHLET_EPS=$DIRICHLET_EPS TEMPERATURE=$TEMPERATURE
  MCTS_BATCH_SIZE=$MCTS_BATCH_SIZE VLOSS_BATCH=$VLOSS_BATCH MAX_PLIES=$MAX_PLIES
  MODEL_BLOCKS=$MODEL_BLOCKS MODEL_FILTERS=$MODEL_FILTERS MODEL_HIDDEN=$MODEL_HIDDEN
  TRAIN_BATCH=$TRAIN_BATCH TRAIN_LR=$TRAIN_LR WEIGHT_SAVE_EVERY=$WEIGHT_SAVE_EVERY
  CHECKPOINT_EVERY=$CHECKPOINT_EVERY MIN_BUFFER_GAMES=$MIN_BUFFER_GAMES
  BUFFER_MAX_GAMES=$BUFFER_MAX_GAMES SEED=$SEED
  EVAL_EVERY_SECONDS=$EVAL_EVERY_SECONDS EVAL_GAMES=$EVAL_GAMES EVAL_SIMS=$EVAL_SIMS EVAL_WORKERS=$EVAL_WORKERS
  TORCH_VERSION=$TORCH_VERSION CHESS_VERSION=$CHESS_VERSION
EOF
    ;;

  1)
    echo "[step 1] sanity (host=$VAST_HOST port=$VAST_PORT)"
    "${SSH[@]}" 'nvidia-smi -L; echo ---; python3 --version; echo ---; \
       python3 -c "import sys; assert sys.version_info >= (3,11), sys.version" \
         || echo "WARN: python3 < 3.11; pick a py311+ image"; \
       echo ---; \
       python3 -c "import torch; print(\"torch\", torch.__version__, \"cuda_avail\", torch.cuda.is_available(), \"cap\", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)" 2>&1 \
         || echo "torch not yet installed"; \
       echo ---; df -h /root | head -2'
    ;;

  2)
    echo "[step 2] install + sync"
    "${SSH[@]}" "pip install --quiet 'torch==${TORCH_VERSION}' --index-url https://download.pytorch.org/whl/cu128 \
              && pip install --quiet 'chess==${CHESS_VERSION}' httpx"
    "${SSH[@]}" "mkdir -p $REMOTE_ENGINE $META_DIR"
    rsync -avz --delete -e "$RSYNC_SSH" \
      --exclude '.venv' \
      --exclude '__pycache__' \
      --exclude '.pytest_cache' \
      --exclude 'weights' \
      --exclude 'runs' \
      --exclude 'games' \
      --exclude 'engine/engine' \
      --exclude '*.pyc' \
      --exclude '.mypy_cache' \
      --exclude '.ruff_cache' \
      "$LOCAL_ENGINE/" "$VAST_USER@$VAST_HOST:$REMOTE_ENGINE/"
    "${SSH[@]}" "pip install --quiet -e $REMOTE_ENGINE"
    # Marker — verify async-coordinator code landed (these symbols are recent).
    "${SSH[@]}" "python3 -c 'from chessckers_engine.selfplay_az_async import run_async_training; from chessckers_engine.replay_buffer import ReplayBuffer; from chessckers_engine.trainer_loop import TrainerLoop; print(\"async marker ok\")'" \
      || { echo "ERROR: async marker check failed — local code did not land on remote"; exit 1; }
    ;;

  4)
    echo "[step 4] launch detached async training (RUN_NAME=$RUN_NAME, RESUME=${RESUME:-0})"
    if [[ "${RESUME:-0}" == "1" ]]; then
      "${SSH[@]}" "mkdir -p $META_DIR && rm -f $META_DIR/exit_code $META_DIR/pid"
      RESUME_FLAG="--base $REMOTE_ENGINE/runs/$RUN_NAME/weights.pt"
      echo "  resume mode: loading from $REMOTE_ENGINE/runs/$RUN_NAME/weights.pt"
    else
      "${SSH[@]}" "mkdir -p $META_DIR && rm -f $META_DIR/exit_code $META_DIR/train.log $META_DIR/pid"
      RESUME_FLAG=""
    fi
    "${SSH[@]}" "cat > $META_DIR/run.sh" <<EOF
#!/usr/bin/env bash
# Strip vast.ai's CUDA forward-compat libcuda — error 804 on cu128 consumer GPUs.
export LD_LIBRARY_PATH="\$(echo \$LD_LIBRARY_PATH | tr ':' '\n' | grep -v '/cuda/compat' | paste -sd: -)"
cd $REMOTE_ENGINE
python3 -m chessckers_engine.selfplay_az_async \\
  --run-dir $REMOTE_ENGINE/runs/$RUN_NAME \\
  --device cuda --workers $WORKERS --sims $SIMS \\
  --d-hidden $MODEL_HIDDEN --c-filters $MODEL_FILTERS --n-blocks $MODEL_BLOCKS \\
  --temperature $TEMPERATURE \\
  --dirichlet-alpha $DIRICHLET_ALPHA --dirichlet-eps $DIRICHLET_EPS \\
  --mcts-batch-size $MCTS_BATCH_SIZE --vloss-batch $VLOSS_BATCH --max-plies $MAX_PLIES \\
  --trainer-batch-size $TRAIN_BATCH --trainer-lr $TRAIN_LR \\
  --weight-save-every $WEIGHT_SAVE_EVERY --checkpoint-every $CHECKPOINT_EVERY \\
  --min-buffer-games $MIN_BUFFER_GAMES --buffer-max-games $BUFFER_MAX_GAMES \\
  --eval-every-seconds $EVAL_EVERY_SECONDS --eval-games $EVAL_GAMES \\
  --eval-sims $EVAL_SIMS --eval-workers $EVAL_WORKERS \\
  --run-seconds $RUN_SECONDS --seed $SEED $RESUME_FLAG
echo \$? > $META_DIR/exit_code
EOF
    ssh -n "${SSH_OPTS[@]}" -p "$VAST_PORT" "$VAST_USER@$VAST_HOST" \
      "chmod +x $META_DIR/run.sh && cd $META_DIR && { nohup ./run.sh </dev/null >train.log 2>&1 & echo \$! > pid && disown && echo launched pid=\$(cat pid); }"
    sleep 3
    "${SSH[@]}" "echo 'pid='\$(cat $META_DIR/pid); echo --- first log lines ---; head -n 12 $META_DIR/train.log 2>/dev/null || echo '(empty)'"
    ;;

  4-status)
    pid="$("${SSH[@]}" "cat $META_DIR/pid 2>/dev/null || echo NONE")"
    if [[ "$pid" == "NONE" ]]; then
      echo "state=not-started"
      exit 0
    fi
    if "${SSH[@]}" "[ -f $META_DIR/exit_code ]"; then
      ec="$("${SSH[@]}" "cat $META_DIR/exit_code")"
      [[ "$ec" == "0" ]] && state=done || state="crashed(ec=$ec)"
    elif "${SSH[@]}" "kill -0 $pid 2>/dev/null"; then
      state=running
    else
      state=dead-no-exit-code
    fi
    echo "state=$state pid=$pid"
    echo "--- log tail (filtered) ---"
    "${SSH[@]}" "tr '\r' '\n' < $META_DIR/train.log 2>/dev/null | grep -E 'step=|EVAL|spawned|seeded|ERROR|Traceback|shut' | tail -n 20 || true"
    echo "--- last eval ---"
    "${SSH[@]}" "tail -n 1 $REMOTE_ENGINE/runs/$RUN_NAME/eval.jsonl 2>/dev/null || echo '(no eval yet)'"
    ;;

  4-wait)
    echo "[step 4-wait] blocking on remote until exit_code lands"
    "${SSH[@]}" "until [ -f $META_DIR/exit_code ]; do sleep 60; done; echo done ec=\$(cat $META_DIR/exit_code)"
    ;;

  5-up)
    echo "[step 5-up] rsync local runs/$RUN_NAME UP to remote"
    if [[ ! -d "$LOCAL_ENGINE/runs/$RUN_NAME" ]]; then
      echo "ERROR: $LOCAL_ENGINE/runs/$RUN_NAME does not exist locally" >&2
      exit 1
    fi
    "${SSH[@]}" "mkdir -p $REMOTE_ENGINE/runs/$RUN_NAME"
    rsync -avz -e "$RSYNC_SSH" \
      "$LOCAL_ENGINE/runs/$RUN_NAME/" \
      "$VAST_USER@$VAST_HOST:$REMOTE_ENGINE/runs/$RUN_NAME/"
    "${SSH[@]}" "ls -lh $REMOTE_ENGINE/runs/$RUN_NAME/weights.pt 2>&1 | head"
    ;;

  5)
    echo "[step 5] rsync runs/$RUN_NAME + train.log back"
    mkdir -p "$LOCAL_ENGINE/runs/$RUN_NAME"
    # Skip the buffer (gigabytes of pickles) — only weights, checkpoints, eval log.
    rsync -avz -e "$RSYNC_SSH" \
      --exclude 'buffer' \
      "$VAST_USER@$VAST_HOST:$REMOTE_ENGINE/runs/$RUN_NAME/" \
      "$LOCAL_ENGINE/runs/$RUN_NAME/"
    rsync -avz -e "$RSYNC_SSH" \
      "$VAST_USER@$VAST_HOST:$META_DIR/train.log" \
      "$LOCAL_ENGINE/runs/$RUN_NAME/train.log" || true
    echo "checkpoints landed in: $LOCAL_ENGINE/runs/$RUN_NAME"
    ;;

  6)
    echo "[step 6] destroy vast.ai instance"
    if [[ -z "${VAST_AI_API_KEY:-}" ]]; then
      VAST_AI_API_KEY="$(grep '^export VAST_AI_API_KEY=' ~/.zshrc 2>/dev/null \
        | sed 's/^export VAST_AI_API_KEY=//; s/^"\(.*\)"$/\1/' || true)"
    fi
    if [[ -z "${VAST_AI_API_KEY:-}" ]]; then
      echo "ERROR: VAST_AI_API_KEY not set and not found in ~/.zshrc" >&2
      exit 3
    fi
    if [[ -z "${VAST_INSTANCE_ID:-}" ]]; then
      VAST_INSTANCE_ID="$(VAST_API_KEY="$VAST_AI_API_KEY" vastai show instances --raw 2>/dev/null \
        | grep -v '^DEPRECATED' \
        | VAST_HOST="$VAST_HOST" python3 -c "
import json, os, sys
host = os.environ['VAST_HOST']
try:
    data = json.load(sys.stdin)
except Exception as e:
    print(f'ERROR: failed to parse vastai output: {e}', file=sys.stderr); sys.exit(2)
matches = [str(i.get('id')) for i in data
           if i.get('public_ipaddr') == host or i.get('ssh_host') == host]
if len(matches) != 1:
    print(f'AMBIGUOUS: {len(matches)} matches for host={host!r}; set VAST_INSTANCE_ID', file=sys.stderr)
    sys.exit(1)
print(matches[0])
")"
    fi
    echo "destroying instance id=$VAST_INSTANCE_ID"
    VAST_API_KEY="$VAST_AI_API_KEY" vastai destroy instance "$VAST_INSTANCE_ID" -y
    ;;

  all)
    STEP=1 "$0"
    STEP=2 "$0" || { echo "step 2 failed; destroying"; STEP=6 "$0" || true; exit 1; }
    STEP=4 "$0" || { echo "step 4 failed; destroying"; STEP=6 "$0" || true; exit 1; }
    STEP=4-wait "$0"
    STEP=5 "$0" || true
    STEP=6 "$0"
    ;;

  *)
    echo "unknown STEP=$STEP" >&2
    exit 2
    ;;
esac
