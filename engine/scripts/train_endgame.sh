#!/usr/bin/env bash
# Single entry point for endgame curriculum self-play training.
#
# One trainer (chessckers_engine.selfplay_az_loop), one output dir
# (engine/weights/run), one log (/tmp/chessckers_train.log). The validated
# config is baked in so there are no ad-hoc invocations:
#   - start from the mate-in-1 seed (the position self-play can bootstrap from)
#   - --sims 400          (enough to concentrate over the ~45-move branching)
#   - --eval-games 0      (eval off; the self-play W/B/D column is the signal)
#   - value/backup length discount γ=0.9 (incentive to win faster)
#
# Everything is overridable via env or pass-through flags, e.g.:
#   CHESSCKERS_START_FEN='<deeper seed>' OUT=weights/run scripts/train_endgame.sh --resume
#   scripts/train_endgame.sh --iterations 50
set -euo pipefail
cd "$(dirname "$0")/.."   # -> engine/

export CHESSCKERS_START_FEN="${CHESSCKERS_START_FEN:-8/8/8/8/8/3kk3/8/4K3[d3:kk,e3:kk] b - - 0 1}"
export CHESSCKERS_MAX_PLIES="${CHESSCKERS_MAX_PLIES:-40}"
export CHESSCKERS_VALUE_DISCOUNT="${CHESSCKERS_VALUE_DISCOUNT:-0.9}"
OUT="${OUT:-weights/run}"
# True-parallel self-play: multiprocess workers, one torch thread each (set in
# the worker via CHESSCKERS_TORCH_THREADS, default 1). Measured ~linear scaling
# to the perf-core count; threads/MPS-batching don't help (GIL-bound feeders,
# CPU forward is compute-bound). Default workers = perf cores; override via WORKERS=.
WORKERS="${WORKERS:-$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 6)}"

echo "start FEN : $CHESSCKERS_START_FEN"
echo "max plies : $CHESSCKERS_MAX_PLIES | discount γ: $CHESSCKERS_VALUE_DISCOUNT | out: $OUT | workers: $WORKERS"

exec .venv/bin/python -m chessckers_engine.selfplay_az_loop \
    --iterations 30 --games-per-iter 8 --sims 400 --epochs 3 \
    --eval-games 0 --workers "$WORKERS" --worker-mode processes --device cpu \
    --weights-dir "$OUT" \
    "$@"
