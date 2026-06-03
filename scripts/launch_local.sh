#!/usr/bin/env bash
# Local-only keep-best self-play loop — ONE box, no Leena / no vast / no HTTP.
#
#   trainer (MPS)  --run-dir run   publishes weights.pt + iter-async ckpts, drains buffer/
#   arena   (CPU)  --run-dir run   seeds best.pt, gates ckpts -> best.pt (imbalance-aware)
#   workers (CPU)  --run-dir run --weights run/best.pt   self-play vs the gated best, write buffer/
#
# Self-play reads the GATED best.pt directly off disk (workers poll its mtime and
# reload on each promotion) — that IS the keep-best signal; no server indirection
# is needed on a single box. All three processes append to $LOG, so it stays one
# stream to watch: [selfplay] [train] [ckpt] [arena].
#
# Usage:
#   scripts/launch_local.sh           # launch (aborts if a fleet process is already up)
#   FRESH=1 scripts/launch_local.sh   # rm -rf run/ first -> completely new random weights
#
# Tunables (env): SIMS(=100) WORKERS(=4) RESIGN(=0.0; set 0.90 to enable) LOG(=/tmp/cc_train.log)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENG="$REPO_ROOT/engine"
PY="$ENG/.venv/bin/python"
RUN="$ENG/weights/run"
LOG="${LOG:-/tmp/cc_train.log}"
SEED_MIX="$REPO_ROOT/scripts/seed_mix.txt"
DH=256; CF=96; NB=4
SIMS="${SIMS:-100}"; WORKERS="${WORKERS:-4}"; RESIGN="${RESIGN:-0.0}"

say(){ echo "[launch-local] $*" >&2; }

if pgrep -f 'chessckers_engine\.(train_continuous|fleet_server|fleet_arena|fleet_client|selfplay_workers_only)' >/dev/null; then
  say "ABORT: a fleet process is already running. Stop it first:  scripts/stop_local.sh"
  exit 1
fi

if [ -n "${FRESH:-}" ]; then
  say "FRESH: rm -rf $RUN  (completely new random weights)"
  rm -rf "$RUN"
fi
mkdir -p "$RUN/buffer"

# Curriculum seeds -> CHESSCKERS_START_FEN (';'-joined; new_game() samples one per game).
SEED_FENS="$(grep -vE '^[[:space:]]*(#|$)' "$SEED_MIX" | paste -sd ';' -)"
say "seed mix: $(grep -cvE '^[[:space:]]*(#|$)' "$SEED_MIX") positions from $SEED_MIX"

# Canonical self-play params -> run/selfplay.json: the ONE source of truth. Local
# workers read it live (per game); fleet_server serves it at /selfplay so leena's
# client mirrors it onto that box — local & leena can no longer drift. Anneal the
# fleet mid-run by editing this file (workers pick it up at the next game boundary,
# no restart); the worker CLI flags below are only the fallback if it goes missing.
cat > "$RUN/selfplay.json" <<JSON
{"sims": $SIMS, "c_puct": 1.5, "temperature": 1.0, "dirichlet_alpha": 0.5, "dirichlet_eps": 0.40, "max_plies": 200}
JSON
say "self-play params -> $RUN/selfplay.json (sims=$SIMS)"

cd "$ENG"

# 1. trainer — random init (no --base), MPS, archive off (matches the prior run).
nohup "$PY" -m chessckers_engine.train_continuous \
  --run-dir "$RUN" --no-prime \
  --buffer-cap 300000 --min-buffer 2000 --replay-factor 8 --batch-size 256 \
  --publish-seconds 45 --ckpt-seconds 120 \
  --d-hidden $DH --c-filters $CF --n-blocks $NB --seed 1000 \
  >>"$LOG" 2>&1 &
say "trainer  pid $!"

# 2. arena — seeds best v0 from weights.pt, then gates each new checkpoint.
nohup "$PY" -m chessckers_engine.fleet_arena \
  --run-dir "$RUN" --seed-mix-file "$SEED_MIX" \
  --d-hidden $DH --c-filters $CF --n-blocks $NB \
  --sims 160 --pairs 4 --threshold 0.55 --side-floor 0.45 \
  --max-plies 200 --gate-seconds 60 --device cpu \
  >>"$LOG" 2>&1 &
say "arena    pid $!"

# 3. workers — self-play vs the gated best.pt (wait for it, reload on promotion).
nohup env CHESSCKERS_START_FEN="$SEED_FENS" MACHINE=local "$PY" -m chessckers_engine.selfplay_workers_only \
  --run-dir "$RUN" --weights "$RUN/best.pt" \
  --workers "$WORKERS" --native --device cpu --sims "$SIMS" \
  --d-hidden $DH --c-filters $CF --n-blocks $NB \
  --max-plies 200 --resign-threshold "$RESIGN" \
  >>"$LOG" 2>&1 &
say "workers  pid $!  ($WORKERS native, sims=$SIMS, resign=$RESIGN)"

say "up. one log to watch:  tail -f $LOG"
