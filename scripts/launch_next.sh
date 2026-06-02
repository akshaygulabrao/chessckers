#!/usr/bin/env bash
# Launch the NEXT local self-play + training run (handoff from the current one).
#
# What it does, in order:
#   1. Stops the currently-running local trainer + workers (clean STOP, then
#      SIGTERM, then SIGKILL after a grace period).
#   2. Snapshots the current net (weights/run/weights.pt) to --base for the new
#      run, and moves the old iter-async checkpoints aside so they're preserved.
#   3. Starts train_continuous (bigger buffer, archive reuse for a WARM start)
#      and the local workers with the new seed mix + 200-ply cap.
#
# Run-dir is reused (weights/run) so leena_sync + leena keep their paths. Leena
# is redeployed separately (it's a different machine) — see the printed hint.
#
#   scripts/launch_next.sh            # do the handoff
#   DRY_RUN=1 scripts/launch_next.sh  # print the commands, change nothing
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENG="$REPO_ROOT/engine"
RUN_DIR="$ENG/weights/run"
ARCHIVE_DIR="/Volumes/hd0/chessckers_archive/run"   # reused -> primes buffer from old games
PY="$ENG/.venv/bin/python"

# ---- run config (edit here) ----
BUFFER_CAP=150000          # bigger replay window (was 50000); RAM-resident, ~225 MB
MIN_BUFFER=2000
ARCHIVE_CAP_GB=380         # FIFO-cap the 400 GB stick (oldest shards evicted)
MAX_PLIES=200              # was 80
SIMS=400
WORKERS=6
DHID=256; CFIL=96; NBLK=4  # arch — UNCHANGED from the current run (2.5 M params)
# --------------------------------

SEED_MIX="$(grep -vE '^\s*#|^\s*$' "$REPO_ROOT/scripts/seed_mix.txt" | paste -sd ';' -)"
[ -n "$SEED_MIX" ] || { echo "empty seed mix (scripts/seed_mix.txt)" >&2; exit 1; }
N_SEEDS=$(grep -cvE '^\s*#|^\s*$' "$REPO_ROOT/scripts/seed_mix.txt")

log() { echo "[$(date +%H:%M:%S)] $*"; }
run() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY: $*"; else eval "$*"; fi; }

log "next run: $N_SEEDS seeds | buffer_cap=$BUFFER_CAP | max_plies=$MAX_PLIES | arch ${DHID}/${CFIL}/${NBLK}"
log "seed mix: $SEED_MIX"

# ---- 1. stop the current run ----
log "stopping current local run (trainer + workers)…"
run "touch '$RUN_DIR/STOP' 2>/dev/null || true"
run "pkill -TERM -f chessckers_engine.train_continuous 2>/dev/null || true"
run "pkill -TERM -f chessckers_engine.selfplay_workers_only 2>/dev/null || true"
if [ "${DRY_RUN:-0}" != 1 ]; then
  for _ in $(seq 30); do
    pgrep -f 'chessckers_engine.(train_continuous|selfplay_workers_only)' >/dev/null || break
    sleep 1
  done
  pkill -9 -f chessckers_engine.train_continuous 2>/dev/null || true
  pkill -9 -f chessckers_engine.selfplay_workers_only 2>/dev/null || true
fi
run "rm -f '$RUN_DIR/STOP'"

# ---- 2. snapshot the current net + preserve old checkpoints ----
BASE="$ENG/weights/base_curriculum_v4.pt"
if [ -f "$RUN_DIR/weights.pt" ]; then
  run "cp '$RUN_DIR/weights.pt' '$BASE'"
  log "snapshotted current net -> $BASE (warm-start --base)"
else
  echo "WARN: no $RUN_DIR/weights.pt to snapshot; new run starts from random init" >&2
  BASE=""
fi
# Move old iter-async checkpoints aside so the new run's numbering doesn't clobber them.
if ls "$RUN_DIR"/iter-async-*.pt >/dev/null 2>&1; then
  PREV="$RUN_DIR/prev_ckpts"
  run "mkdir -p '$PREV' && mv '$RUN_DIR'/iter-async-*.pt '$PREV'/ 2>/dev/null || true"
  log "moved old checkpoints -> $PREV"
fi

# ---- 3. start the new run ----
BASE_ARG=""; [ -n "$BASE" ] && BASE_ARG="--base '$BASE'"
log "starting trainer (train_continuous)…"
run "cd '$ENG' && nohup '$PY' -m chessckers_engine.train_continuous \
  --run-dir '$RUN_DIR' $BASE_ARG \
  --archive-dir '$ARCHIVE_DIR' --archive-cap-gb $ARCHIVE_CAP_GB \
  --buffer-cap $BUFFER_CAP --min-buffer $MIN_BUFFER \
  --replay-factor 8 --batch-size 256 --lr 1e-3 \
  --publish-seconds 45 --ckpt-seconds 120 \
  --device auto --d-hidden $DHID --c-filters $CFIL --n-blocks $NBLK \
  > '$RUN_DIR/trainer.log' 2>&1 & echo \$! > '$RUN_DIR/trainer.pid'"

sleep 2  # let the trainer publish initial weights.pt before workers poll
log "starting local workers (new seed mix, ${MAX_PLIES}-ply cap)…"
run "cd '$ENG' && CHESSCKERS_START_FEN='$SEED_MIX' CHESSCKERS_MAX_PLIES=$MAX_PLIES \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MACHINE=local \
  nohup '$PY' -m chessckers_engine.selfplay_workers_only \
  --run-dir '$RUN_DIR' --weights '$RUN_DIR/weights.pt' \
  --workers $WORKERS --worker-id-base 0 --device cpu --sims $SIMS \
  --d-hidden $DHID --c-filters $CFIL --n-blocks $NBLK \
  --temperature 1.0 --dirichlet-alpha 0.5 --dirichlet-eps 0.40 \
  --max-plies $MAX_PLIES --weights-poll-seconds 20 --seed 1000 \
  > '$RUN_DIR/workers.log' 2>&1 & echo \$! > '$RUN_DIR/workers.pid'"

log "done. follow: tail -f $RUN_DIR/trainer.log"
log "leena: redeploy with the new mix + 200-ply cap, e.g."
log "  scp scripts/seed_mix.txt engine/scripts/leena_launch.sh leena:~/chessckers/  && ssh leena 'bash ~/chessckers/leena_launch.sh'"
