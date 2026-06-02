#!/usr/bin/env bash
# Launch the NEXT self-play + training run in its OWN run-dir, warm-started from
# the previous (now-finished) run's net.
#
# Assumes the current run has already FINISHED (you're waiting for it). It will
# ABORT if a trainer is still active — it does NOT kill the current run.
#
# What it does:
#   1. Guard: refuse to start while a train_continuous is still running.
#   2. Snapshot the previous run's net (weights/run/weights.pt) -> --base.
#   3. Start train_continuous + local workers in a FRESH run-dir (RUN_NAME),
#      with the new seed mix, 200-ply cap, and a 300k replay window. The archive
#      is reused (--archive-dir) so the buffer PRIMES from the old games while
#      the new mix collects.
#
#   scripts/launch_next.sh            # start the new run
#   DRY_RUN=1 scripts/launch_next.sh  # print the commands, change nothing
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENG="$REPO_ROOT/engine"
PY="$ENG/.venv/bin/python"

# ---- run config (edit here) ----
RUN_NAME=run-frontier      # the new run's name -> engine/weights/$RUN_NAME
PREV_RUN_DIR="$ENG/weights/run"                     # finished run; source of the warm-start net
ARCHIVE_DIR="/Volumes/hd0/chessckers_archive/run"   # reused -> primes the buffer from old games
BUFFER_CAP=300000          # replay window (was 50000); RAM-resident, ~450 MB
MIN_BUFFER=2000
ARCHIVE_CAP_GB=380         # FIFO-cap the 400 GB stick (oldest shards evicted)
MAX_PLIES=200              # was 80
SIMS=400
WORKERS=6
DHID=256; CFIL=96; NBLK=4  # arch — UNCHANGED from the current run (2.5 M params)
# --------------------------------

RUN_DIR="$ENG/weights/$RUN_NAME"
SEED_MIX="$(grep -vE '^\s*#|^\s*$' "$REPO_ROOT/scripts/seed_mix.txt" | paste -sd ';' -)"
[ -n "$SEED_MIX" ] || { echo "empty seed mix (scripts/seed_mix.txt)" >&2; exit 1; }
N_SEEDS=$(grep -cvE '^\s*#|^\s*$' "$REPO_ROOT/scripts/seed_mix.txt")

log() { echo "[$(date +%H:%M:%S)] $*"; }
run() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY: $*"; else eval "$*"; fi; }

# ---- guard: don't start over a live run ----
if [ "${DRY_RUN:-0}" != 1 ] && pgrep -f 'chessckers_engine.train_continuous' >/dev/null; then
  echo "ABORT: a train_continuous is still running. Wait for the current run to finish" >&2
  echo "       (or stop it) before launching '$RUN_NAME'." >&2
  exit 1
fi

log "next run '$RUN_NAME': $N_SEEDS seeds | buffer_cap=$BUFFER_CAP | max_plies=$MAX_PLIES | arch ${DHID}/${CFIL}/${NBLK}"
log "seed mix: $SEED_MIX"
run "mkdir -p '$RUN_DIR/buffer'"

# ---- snapshot the previous run's net -> --base ----
BASE="$ENG/weights/base_curriculum_v4.pt"
if [ -f "$PREV_RUN_DIR/weights.pt" ]; then
  run "cp '$PREV_RUN_DIR/weights.pt' '$BASE'"
  log "warm-start net: $PREV_RUN_DIR/weights.pt -> $BASE"
else
  echo "WARN: no $PREV_RUN_DIR/weights.pt to warm-start from; new run starts from random init" >&2
  BASE=""
fi

# ---- start the new run ----
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
log "leena (redeploy + start the sync sidecar for THIS run-dir):"
log "  scp scripts/seed_mix.txt engine/scripts/leena_launch.sh leena:~/chessckers/ && ssh leena 'bash ~/chessckers/leena_launch.sh'"
log "  RUN_DIR='$RUN_DIR' nohup bash engine/scripts/leena_sync.sh >/tmp/cc_leena_sync.log 2>&1 &"
