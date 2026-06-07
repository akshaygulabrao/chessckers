#!/usr/bin/env bash
# A/B the architecture redesign at MATCHED parameter count: train a V1 net (2.51M,
# pooled scorer, ResNet trunk) and a V2 net (2.52M, square-grounded gather head +
# residual-first Transformer trunk / ResTNet RRT) under an IDENTICAL self-play +
# training budget and seed, then gauntlet them head-to-head (challenger=V2 vs
# champion=V1). The single question: at the same param budget, does the new
# architecture (gather head + attention) beat the old one?
#
# The v2 arm defaults to the transformer config that lands at ~2.52M params
# (9 residual + 7 transformer blocks @ 96 filters) — the +9.6K over V1 is exactly
# the learned positional embedding. Set V2_TF_BLOCKS=0 to A/B the gather head
# ALONE (the leaner 899K V2, isolating the head change from the transformer/scale).
#
# Knobs (env): ITERS GAMES SIMS SEED WORKERS  +  GGAMES GSIMS (gauntlet)
#              V2_BLOCKS V2_TF_BLOCKS V2_HEADS V2_FF (v2 trunk shape).
# Both runs are independent — comment one out to resume/redo just the other.
# Checkpoints self-describe (best.pt + best.pt.arch.json), so the gauntlet rebuilds
# each side's exact trunk automatically; the --*-version flags are just fallbacks.
set -euo pipefail
cd "$(dirname "$0")/.."                       # engine/
PY=.venv/bin/python

# Both trainings AND the gauntlet start from this position (PyVariantClient.new_game
# reads CHESSCKERS_START_FEN). Default = the "3 two-king stacks vs 8 pawns" curriculum
# position (Black towers d6/e6/f6 each kk; White's 8 pawns on rank 2; Black to move) —
# a constrained, fast-signal board to test whether V2 outlearns V1. Override to taste.
export CHESSCKERS_START_FEN="${CHESSCKERS_START_FEN:-8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1}"
export CHESSCKERS_MAX_PLIES="${CHESSCKERS_MAX_PLIES:-200}"

ITERS="${ITERS:-20}"; GAMES="${GAMES:-24}"; SIMS="${SIMS:-200}"; SEED="${SEED:-0}"
WORKERS="${WORKERS:-$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 4)}"
GGAMES="${GGAMES:-40}"; GSIMS="${GSIMS:-200}"
# v2 transformer trunk: 9 residual + 7 transformer @ 96 filters, 4 heads → 2.52M.
V2_BLOCKS="${V2_BLOCKS:-9}"; V2_TF_BLOCKS="${V2_TF_BLOCKS:-7}"
V2_HEADS="${V2_HEADS:-4}"; V2_FF="${V2_FF:-4}"
echo "start FEN : $CHESSCKERS_START_FEN"

run() {  # $1 = v1|v2 ; $2.. = extra arch flags
  local ver="$1"; shift
  echo "=== train $ver (iters=$ITERS games=$GAMES sims=$SIMS seed=$SEED) ==="
  "$PY" -m chessckers_engine.selfplay_az_loop \
    --arch-version "$ver" --seed "$SEED" \
    --iterations "$ITERS" --games-per-iter "$GAMES" --sims "$SIMS" \
    --workers "$WORKERS" --weights-dir "weights/ab-$ver" --keep-best "$@"
}

run v1
run v2 --model-blocks "$V2_BLOCKS" --tf-blocks "$V2_TF_BLOCKS" \
       --tf-heads "$V2_HEADS" --tf-ff-mult "$V2_FF"

echo "=== gauntlet: V2 (challenger) vs V1 (champion) ==="
"$PY" -m chessckers_engine.gauntlet \
  --challenger weights/ab-v2/best.pt --challenger-version v2 \
  --champion  weights/ab-v1/best.pt --champion-version  v1 \
  --games "$GGAMES" --sims "$GSIMS"
