#!/usr/bin/env bash
# Train AlphaZero self-play from the "4-stone tower vs lone White king" endgame
# instead of the full starting position. This is a Black-competence detector:
# with best play Black (the e6 tower) should hunt down and capture the lone king,
# so a genuinely strong Black net should win ~100% of these games. Watch the
# per-iteration eval (net-as-black vs random-white) climb toward all-wins.
#
# The custom start position + short ply cap are injected via env vars read by
# PyVariantClient.new_game() and play_az_game() — every self-play/eval game
# (which all go through new_game()) starts from this FEN.
#
# Usage:
#   scripts/train_endgame.sh                 # sensible defaults
#   scripts/train_endgame.sh --iterations 50 # override / add any loop flag
set -euo pipefail
cd "$(dirname "$0")/.."   # -> engine/

export CHESSCKERS_START_FEN="${CHESSCKERS_START_FEN:-8/8/4p3/8/8/8/8/4K3[e6:ssss] b - - 0 1}"
export CHESSCKERS_MAX_PLIES="${CHESSCKERS_MAX_PLIES:-80}"

echo "start FEN : $CHESSCKERS_START_FEN"
echo "max plies : $CHESSCKERS_MAX_PLIES"

exec .venv/bin/python -m chessckers_engine.selfplay_az_loop \
    --iterations 30 --games-per-iter 16 --sims 400 --epochs 3 \
    --eval-games 0 --workers 1 --worker-mode threads --device auto \
    --weights-dir weights/endgame \
    "$@"
