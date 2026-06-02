#!/usr/bin/env bash
# Launch the NEXT self-play + training run, reusing the weights/run dir,
# warm-started from the previous (now-finished) run's net.
#
# Assumes the current run has already FINISHED (you're waiting for it). It will
# ABORT if a trainer is still active — it does NOT kill the current run.
#
# What it does:
#   1. Guard: refuse to start while a train_continuous is still running.
#   2. Snapshot weights/run/weights.pt -> base_curriculum_v4.pt (--base) and
#      move the finished run's iter-async checkpoints aside (prev_ckpts/) so the
#      new run's numbering doesn't clobber them.
#   3. Start train_continuous + local workers in weights/run, with the new seed
#      mix, 200-ply cap, and a 300k replay window. The archive is reused
#      (--archive-dir) so the buffer PRIMES from the old games while the new mix
#      collects.
#   4. Best-effort bring up leena: if reachable, deploy the seed mix + relaunch
#      its workers and start the local leena_sync sidecar. If leena is asleep /
#      off-network, warn and continue LOCAL-ONLY (never aborts the run).
#
# All three writers (trainer, workers, leena_sync) share ONE log: /tmp/cc_train.log
# (truncated fresh at each launch). Follow the whole run with: tail -f /tmp/cc_train.log
#
#   scripts/launch_next.sh            # start the new run (local + leena)
#   SKIP_LEENA=1 scripts/launch_next.sh   # local only
#   DRY_RUN=1 scripts/launch_next.sh  # print the commands, change nothing
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENG="$REPO_ROOT/engine"
PY="$ENG/.venv/bin/python"

# ---- run config (edit here) ----
RUN_DIR="$ENG/weights/run"                          # reused across runs
LOG=/tmp/cc_train.log                               # SINGLE unified log: trainer + workers + leena_sync
ARCHIVE_DIR="/Volumes/hd0/chessckers_archive/run"   # reused -> primes the buffer from old games
BUFFER_CAP=300000          # replay window (was 50000); RAM-resident, ~450 MB
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

# ---- guard: don't start over a live run ----
if [ "${DRY_RUN:-0}" != 1 ] && pgrep -f 'chessckers_engine.train_continuous' >/dev/null; then
  echo "ABORT: a train_continuous is still running. Wait for the current run to finish" >&2
  echo "       (or stop it) before launching the next run." >&2
  exit 1
fi

log "next run (weights/run): $N_SEEDS seeds | buffer_cap=$BUFFER_CAP | max_plies=$MAX_PLIES | arch ${DHID}/${CFIL}/${NBLK}"
log "seed mix: $SEED_MIX"
run "mkdir -p '$RUN_DIR/buffer'"
run ": > '$LOG'"   # fresh unified log for this run
log "unified log -> $LOG"

# ---- snapshot the finished run's net -> --base, preserve its checkpoints ----
BASE="$ENG/weights/base_curriculum_v4.pt"
if [ -f "$RUN_DIR/weights.pt" ]; then
  run "cp '$RUN_DIR/weights.pt' '$BASE'"
  log "warm-start net: $RUN_DIR/weights.pt -> $BASE"
else
  echo "WARN: no $RUN_DIR/weights.pt to warm-start from; new run starts from random init" >&2
  BASE=""
fi
if ls "$RUN_DIR"/iter-async-*.pt >/dev/null 2>&1; then
  PREV="$RUN_DIR/prev_ckpts"
  run "mkdir -p '$PREV' && mv '$RUN_DIR'/iter-async-*.pt '$PREV'/ 2>/dev/null || true"
  log "preserved finished-run checkpoints -> $PREV"
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
  >> '$LOG' 2>&1 & echo \$! > '$RUN_DIR/trainer.pid'"

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
  >> '$LOG' 2>&1 & echo \$! > '$RUN_DIR/workers.pid'"

# ---- leena (best-effort: never abort the local run) ----
LEENA="${LEENA:-leenagulabrao@Leenas-MacBook-Air.local}"   # Bonjour — survives DHCP changes
SSH_LEENA="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
LSYNC="$ENG/scripts/leena_sync.sh"
leena_manual="scp '$REPO_ROOT/scripts/seed_mix.txt' '$ENG/scripts/leena_launch.sh' '$LEENA:chessckers/' && $SSH_LEENA '$LEENA' 'bash ~/chessckers/leena_launch.sh' && nohup bash '$LSYNC' >>'$LOG' 2>&1 &"

if [ "${SKIP_LEENA:-0}" = 1 ]; then
  log "leena: skipped (SKIP_LEENA=1)"
elif [ "${DRY_RUN:-0}" = 1 ]; then
  echo "DRY: (if reachable) scp seed_mix.txt + leena_launch.sh -> leena; ssh leena bash leena_launch.sh; start leena_sync.sh"
elif $SSH_LEENA "$LEENA" true 2>/dev/null; then
  log "leena: reachable — deploying seed mix + (re)launching workers"
  if scp -q "$REPO_ROOT/scripts/seed_mix.txt" "$ENG/scripts/leena_launch.sh" "$LEENA:chessckers/" 2>/dev/null \
     && $SSH_LEENA "$LEENA" 'pkill -f selfplay_workers_only 2>/dev/null; pkill -f multiprocessing.spawn 2>/dev/null; sleep 1; bash ~/chessckers/leena_launch.sh'; then
    pkill -f "$LSYNC" 2>/dev/null || true              # drop any stale sidecar from the prior run
    nohup bash "$LSYNC" >>"$LOG" 2>&1 &
    log "leena: workers launched + sync sidecar started (pid $!)"
  else
    log "leena: deploy/launch FAILED — continuing local-only. Retry: $leena_manual"
  fi
else
  log "leena: UNREACHABLE (asleep / off-network?) — started local-only."
  log "       once it's up:  $leena_manual"
fi

log "done. follow: tail -f $LOG"
