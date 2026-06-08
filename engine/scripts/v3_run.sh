#!/usr/bin/env bash
# Train V3 (2.52M ResNet+Transformer hybrid, gather head) under the SAME conditions
# as the V1/V2 A/B, then a 3-way round-robin gauntlet (temperature 0.3) over:
#   V1 = 2.5M pooled-ResNet (weights/ab-v1/best.pt)
#   V2 = 899K gather-head ResNet (weights/ab-v2/best.pt)
#   V3 = 2.52M gather-head ResNet+Transformer (weights/ab-v3/best.pt, trained here)
# V1/V2 reload via default-arch fallback (no sidecar); V3 via its .arch.json sidecar.
set -euo pipefail
cd "$(dirname "$0")/.."                       # engine/
PY=.venv/bin/python

export CHESSCKERS_START_FEN="${CHESSCKERS_START_FEN:-8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1}"
export CHESSCKERS_MAX_PLIES="${CHESSCKERS_MAX_PLIES:-200}"
ITERS="${ITERS:-20}"; GAMES="${GAMES:-24}"; SIMS="${SIMS:-200}"; SEED="${SEED:-0}"
WORKERS="${WORKERS:-$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 4)}"
GGAMES="${GGAMES:-40}"; GSIMS="${GSIMS:-200}"; GTEMP="${GTEMP:-0.3}"

echo "=== train V3 (2.52M: 9 residual + 7 transformer, gather head) ==="
"$PY" -m chessckers_engine.selfplay_az_loop \
  --arch-version v2 --model-blocks 9 --tf-blocks 7 --tf-heads 4 --tf-ff-mult 4 \
  --seed "$SEED" --iterations "$ITERS" --games-per-iter "$GAMES" --sims "$SIMS" \
  --workers "$WORKERS" --weights-dir weights/ab-v3 --keep-best \
  --native --native-gpu --native-batch-size "${NBATCH:-$GAMES}" --native-concurrency "${NCONC:-0}"

g() {  # $1 chal-name $2 chal-path $3 chal-ver  $4 champ-name $5 champ-path $6 champ-ver
  echo "===== $1 (challenger) vs $4 (champion) — temp $GTEMP ====="
  "$PY" -m chessckers_engine.gauntlet \
    --challenger "$2" --challenger-version "$3" \
    --champion "$5" --champion-version "$6" \
    --games "$GGAMES" --sims "$GSIMS" --temperature "$GTEMP" --max-plies "$CHESSCKERS_MAX_PLIES"
}

echo; echo "########## 3-WAY GAUNTLET ##########"
g V2 weights/ab-v2/best.pt v2  V1 weights/ab-v1/best.pt v1
g V3 weights/ab-v3/best.pt v2  V1 weights/ab-v1/best.pt v1
g V3 weights/ab-v3/best.pt v2  V2 weights/ab-v2/best.pt v2
echo "########## done — score is the CHALLENGER's (left name) in each block ##########"
