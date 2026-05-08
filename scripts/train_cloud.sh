#!/usr/bin/env bash
# scripts/train_cloud.sh
#
# Vast.ai cloud training driver for the Chessckers AlphaZero loop.
# All phases are gated by the STEP env var so each can run independently
# (or chain them via STEP=all).
#
# Usage examples:
#   SSH_STRING="ssh -p 22080 root@1.2.3.4" STEP=1 ./scripts/train_cloud.sh
#   SSH_STRING="ssh -p 22080 root@1.2.3.4" STEP=all ./scripts/train_cloud.sh
#   STEP=4-status VAST_HOST=1.2.3.4 VAST_PORT=22080 ./scripts/train_cloud.sh

set -euo pipefail

STEP="${STEP:-help}"

# === SSH target (skip parse for help) ===
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

# === pins (match local venv so any local source changes stay compatible) ===
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
CHESS_VERSION="${CHESS_VERSION:-1.11.2}"

# === run config (override via env) ===
RUN_NAME="${RUN_NAME:-cloud-run-001}"
ITERATIONS="${ITERATIONS:-100}"
GAMES_PER_ITER="${GAMES_PER_ITER:-200}"
# sims=400 (vs AZ-chess's 800): 4× deeper search than the default-100 run.
# Spends more compute on each self-play move so MCTS finds longer-horizon
# tactical sequences — the lever for breaking out of the White-favoring basin.
SIMS="${SIMS:-400}"
EVAL_GAMES="${EVAL_GAMES:-10}"
EVAL_SIMS="${EVAL_SIMS:-200}"
WORKERS="${WORKERS:-16}"
VLOSS_BATCH="${VLOSS_BATCH:-8}"
# Without --mcts-batch-size>1 the InferenceServer is bypassed and every leaf
# is evaluated singleton — GPU sits at ~1% util. Default to WORKERS*VLOSS_BATCH
# so the server's max batch matches the theoretical ceiling.
MCTS_BATCH_SIZE="${MCTS_BATCH_SIZE:-$((WORKERS * VLOSS_BATCH))}"
BUFFER_ITERS="${BUFFER_ITERS:-30}"
EPOCHS="${EPOCHS:-5}"
TRAIN_BATCH="${TRAIN_BATCH:-64}"
TEMP="${TEMP:-1.0}"
# Higher final temperature (was 0.2) keeps move sampling stochastic late
# in the game so we explore more diverse endgame positions.
TEMP_FINAL="${TEMP_FINAL:-0.5}"
# Bumped Dirichlet noise (was alpha=0.3 eps=0.25). Wider opening prior
# perturbation so MCTS root sometimes commits to moves the policy underweights.
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.5}"
DIRICHLET_EPS="${DIRICHLET_EPS:-0.40}"
# AZ-chess scale defaults: 20 blocks × 256 filters × 384 hidden = ~30M params.
# Chessckers may be more complex than chess, so this is a floor not a ceiling.
MODEL_BLOCKS="${MODEL_BLOCKS:-20}"
MODEL_FILTERS="${MODEL_FILTERS:-256}"
MODEL_HIDDEN="${MODEL_HIDDEN:-384}"
KEEP_BEST_THRESHOLD="${KEEP_BEST_THRESHOLD:-0.45}"
KEEP_BEST_GAMES="${KEEP_BEST_GAMES:-20}"
SEED="${SEED:-1}"

case "$STEP" in
  help|"")
    cat <<EOF
Usage: STEP=<step> SSH_STRING='ssh -p PORT root@HOST' $0

Steps:
  1          Sanity (GPU, python, disk)
  2          Install torch+chess pinned to local, rsync engine code, pip install -e
  4          Launch training (RESUME=1 reuses weights/<RUN_NAME>/state.json)
  4-status   Token-cheap status: state + last few log lines
  4-wait     Block ON THE REMOTE until exit_code lands (run with run_in_background)
  5          Rsync weights/<RUN_NAME> + train.log back
  5-up       Rsync local weights/<RUN_NAME> UP to remote (for resuming on a fresh box)
  6          Destroy vast.ai instance (auto-detects ID from VAST_HOST)
  all        1 → 2 → 4 → 4-wait → 5 → 6 (destroy is unconditional)

Spot/interruptible recovery:
  After a preemption, rent a new box, rsync local checkpoints back up, resume:
    STEP=2 ./scripts/train_cloud.sh   # install + sync code on new box
    STEP=5-up ./scripts/train_cloud.sh    # push weights/<RUN_NAME> to new box
    RESUME=1 STEP=4 ./scripts/train_cloud.sh   # continue from last completed iter

Tunables (env overrides; defaults shown):
  RUN_NAME=$RUN_NAME ITERATIONS=$ITERATIONS GAMES_PER_ITER=$GAMES_PER_ITER SIMS=$SIMS
  EVAL_GAMES=$EVAL_GAMES EVAL_SIMS=$EVAL_SIMS WORKERS=$WORKERS VLOSS_BATCH=$VLOSS_BATCH MCTS_BATCH_SIZE=$MCTS_BATCH_SIZE
  DIRICHLET_ALPHA=$DIRICHLET_ALPHA DIRICHLET_EPS=$DIRICHLET_EPS
  BUFFER_ITERS=$BUFFER_ITERS EPOCHS=$EPOCHS TRAIN_BATCH=$TRAIN_BATCH
  TEMP=$TEMP TEMP_FINAL=$TEMP_FINAL
  MODEL_BLOCKS=$MODEL_BLOCKS MODEL_FILTERS=$MODEL_FILTERS MODEL_HIDDEN=$MODEL_HIDDEN
  KEEP_BEST_THRESHOLD=$KEEP_BEST_THRESHOLD KEEP_BEST_GAMES=$KEEP_BEST_GAMES SEED=$SEED
  TORCH_VERSION=$TORCH_VERSION CHESS_VERSION=$CHESS_VERSION
EOF
    ;;

  1)
    echo "[step 1] sanity (host=$VAST_HOST port=$VAST_PORT)"
    "${SSH[@]}" 'nvidia-smi -L; echo ---; python3 --version; echo ---; \
       python3 -c "import sys; assert sys.version_info >= (3,11), sys.version" \
         || echo "WARN: python3 < 3.11; pick a py311+ image (e.g. vastai/pytorch:cuda-12.4.1-py311-22.04-cudnn)"; \
       echo ---; \
       python3 -c "import torch; print(\"torch\", torch.__version__, \"cuda_avail\", torch.cuda.is_available(), \"cap\", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)" 2>&1 \
         || echo "torch not yet installed"; \
       echo ---; df -h /root | head -2'
    ;;

  2)
    echo "[step 2] install + sync"
    # 2a: install pinned torch (cu128 wheels — torch 2.11.x is on cu128, not cu121) + chess
    "${SSH[@]}" "pip install --quiet 'torch==${TORCH_VERSION}' --index-url https://download.pytorch.org/whl/cu128 \
              && pip install --quiet 'chess==${CHESS_VERSION}' httpx"
    # 2b: rsync engine code (skip heavy/local-only dirs)
    "${SSH[@]}" "mkdir -p $REMOTE_ENGINE $META_DIR"
    rsync -avz --delete -e "$RSYNC_SSH" \
      --exclude '.venv' \
      --exclude '__pycache__' \
      --exclude '.pytest_cache' \
      --exclude 'weights' \
      --exclude 'games' \
      --exclude 'engine/engine' \
      --exclude '*.pyc' \
      --exclude '.mypy_cache' \
      --exclude '.ruff_cache' \
      "$LOCAL_ENGINE/" "$VAST_USER@$VAST_HOST:$REMOTE_ENGINE/"
    # 2c: editable install
    "${SSH[@]}" "pip install --quiet -e $REMOTE_ENGINE"
    # 2d: marker — verify the local source landed (InferenceServer + keep-best gating are recent)
    "${SSH[@]}" "python3 -c 'from chessckers_engine.inference_server import InferenceServer; from chessckers_engine.selfplay_az_loop import _gate_against_best; print(\"marker ok\")'" \
      || { echo "ERROR: marker check failed — local code did not land on remote"; exit 1; }
    ;;

  4)
    echo "[step 4] launch detached training (RUN_NAME=$RUN_NAME, RESUME=${RESUME:-0})"
    # On RESUME=1: keep existing weights/<RUN_NAME>/ on the remote (state.json
    # tells the trainer where to pick up). Otherwise wipe to start fresh.
    if [[ "${RESUME:-0}" == "1" ]]; then
      RESUME_FLAG="--resume"
      "${SSH[@]}" "mkdir -p $META_DIR && rm -f $META_DIR/exit_code $META_DIR/pid"
      echo "  resume mode: preserving $REMOTE_ENGINE/weights/$RUN_NAME"
    else
      RESUME_FLAG=""
      "${SSH[@]}" "mkdir -p $META_DIR && rm -f $META_DIR/exit_code $META_DIR/train.log $META_DIR/pid"
    fi
    # Pipe a here-doc into ssh's stdin to write run.sh on the remote.
    # Local vars expand; \$? stays literal so it evaluates on remote.
    "${SSH[@]}" "cat > $META_DIR/run.sh" <<EOF
#!/usr/bin/env bash
# Some vast.ai hosts ship a CUDA forward-compat libcuda that breaks consumer
# GPUs ("Error 804: forward compatibility was attempted on non supported HW").
# Drop it from LD_LIBRARY_PATH so torch falls back to the host driver's libcuda.
export LD_LIBRARY_PATH="\$(echo \$LD_LIBRARY_PATH | tr ':' '\n' | grep -v '/cuda/compat' | paste -sd: -)"
cd $REMOTE_ENGINE
python3 -m chessckers_engine.selfplay_az_loop --use-pyvariant --device cuda --iterations $ITERATIONS --games-per-iter $GAMES_PER_ITER --sims $SIMS --workers $WORKERS --vloss-batch $VLOSS_BATCH --mcts-batch-size $MCTS_BATCH_SIZE --buffer-iters $BUFFER_ITERS --epochs $EPOCHS --train-batch-size $TRAIN_BATCH --temperature $TEMP --temperature-final $TEMP_FINAL --dirichlet-alpha $DIRICHLET_ALPHA --dirichlet-eps $DIRICHLET_EPS --eval-games $EVAL_GAMES --eval-sims $EVAL_SIMS --keep-best --keep-best-threshold $KEEP_BEST_THRESHOLD --keep-best-games $KEEP_BEST_GAMES --model-blocks $MODEL_BLOCKS --model-filters $MODEL_FILTERS --model-hidden $MODEL_HIDDEN --weights-dir $REMOTE_ENGINE/weights/$RUN_NAME --seed $SEED $RESUME_FLAG
echo \$? > $META_DIR/exit_code
EOF
    # -n closes stdin so SSH doesn't read from us; </dev/null on the nohup detaches the
    # remote process's stdin too. Without these, SSH hangs waiting on the backgrounded
    # job's inherited fds even after disown.
    # Brace group wraps the bg+echo so `cd` runs in our shell (not inside the backgrounded
    # subshell, which would orphan `pid` to whatever cwd ssh started in).
    ssh -n "${SSH_OPTS[@]}" -p "$VAST_PORT" "$VAST_USER@$VAST_HOST" \
      "chmod +x $META_DIR/run.sh && cd $META_DIR && { nohup ./run.sh </dev/null >train.log 2>&1 & echo \$! > pid && disown && echo launched pid=\$(cat pid); }"
    sleep 3
    "${SSH[@]}" "echo 'pid='\$(cat $META_DIR/pid); echo --- first log lines ---; head -n 8 $META_DIR/train.log 2>/dev/null || echo '(empty)'"
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
    "${SSH[@]}" "tr '\r' '\n' < $META_DIR/train.log 2>/dev/null | grep -E 'iter |eval |gating|ACCEPT|REJECT|ERROR|Traceback|loss=' | tail -n 12 || true"
    ;;

  4-wait)
    echo "[step 4-wait] blocking on remote until exit_code lands"
    "${SSH[@]}" "until [ -f $META_DIR/exit_code ]; do sleep 30; done; echo done ec=\$(cat $META_DIR/exit_code)"
    ;;

  5-up)
    echo "[step 5-up] rsync local weights/$RUN_NAME UP to remote (for spot resume)"
    if [[ ! -d "$LOCAL_ENGINE/weights/$RUN_NAME" ]]; then
      echo "ERROR: $LOCAL_ENGINE/weights/$RUN_NAME does not exist locally" >&2
      exit 1
    fi
    "${SSH[@]}" "mkdir -p $REMOTE_ENGINE/weights/$RUN_NAME"
    rsync -avz -e "$RSYNC_SSH" \
      "$LOCAL_ENGINE/weights/$RUN_NAME/" \
      "$VAST_USER@$VAST_HOST:$REMOTE_ENGINE/weights/$RUN_NAME/"
    "${SSH[@]}" "ls -lh $REMOTE_ENGINE/weights/$RUN_NAME/state.json $REMOTE_ENGINE/weights/$RUN_NAME/best.pt 2>&1 | head"
    ;;

  5)
    echo "[step 5] rsync weights/$RUN_NAME + train.log back"
    mkdir -p "$LOCAL_ENGINE/weights/$RUN_NAME"
    rsync -avz -e "$RSYNC_SSH" \
      "$VAST_USER@$VAST_HOST:$REMOTE_ENGINE/weights/$RUN_NAME/" \
      "$LOCAL_ENGINE/weights/$RUN_NAME/"
    rsync -avz -e "$RSYNC_SSH" \
      "$VAST_USER@$VAST_HOST:$META_DIR/train.log" \
      "$LOCAL_ENGINE/weights/$RUN_NAME/train.log" || true
    echo "checkpoints landed in: $LOCAL_ENGINE/weights/$RUN_NAME"
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
      # vastai prints a deprecation banner before the JSON; strip it.
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
    STEP=5 "$0" || true   # don't skip destroy if rsync fails
    STEP=6 "$0"
    ;;

  *)
    echo "unknown STEP=$STEP" >&2
    exit 2
    ;;
esac
