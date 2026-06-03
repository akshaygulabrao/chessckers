#!/usr/bin/env bash
# Launch leena's self-play workers that feed the local trainer via
# scripts/leena_sync.sh. The seed mix is read from ~/chessckers/seed_mix.txt
# (scp'd alongside this script) — the same canonical file the local launcher
# uses, so local + leena always self-play the SAME curriculum.
# Arch 256/96/4 MUST match the trainer; worker-id-base 300 -> games attribute
# to "leena"; per-worker CPU mode hot-reloads weights pushed by the sync.
cd ~/chessckers/engine
export MACHINE=leena CHESSCKERS_MAX_PLIES=200 CHESSCKERS_VALUE_DISCOUNT=0.98 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
SEED_MIX_FILE="${SEED_MIX_FILE:-$HOME/chessckers/seed_mix.txt}"
[ -f "$SEED_MIX_FILE" ] || { echo "missing seed mix: $SEED_MIX_FILE" >&2; exit 1; }
export CHESSCKERS_START_FEN="$(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$SEED_MIX_FILE" | paste -sd ';' -)"
[ -n "$CHESSCKERS_START_FEN" ] || { echo "empty seed mix in $SEED_MIX_FILE" >&2; exit 1; }
mkdir -p "$HOME/chessckers/run"
# Keep the Air awake with a STANDALONE detached caffeinate so it doesn't idle/lid
# sleep and drop off the network (root cause of leena going unreachable). A
# caffeinate that WRAPS the python did NOT survive ssh-session teardown; a
# standalone one does. Needs leena on AC power.
pkill -x caffeinate 2>/dev/null || true
nohup caffeinate -ims >/dev/null 2>&1 </dev/null & disown
# Prefer the native C++ search engine (same as local). Build it if the
# extension isn't installed here; fall back to the Python engine if the build
# (or import) fails, so leena keeps producing games either way.
NATIVE=""
if ! .venv/bin/python -c "import chessckers_cpp" 2>/dev/null && [ -x cpp/build.sh ]; then
  echo "leena: chessckers_cpp not installed -> building (cpp/build.sh)…"
  cpp/build.sh > "$HOME/chessckers/run/cpp_build.log" 2>&1 \
    || echo "leena: native build FAILED (see run/cpp_build.log) -> Python fallback"
fi
if .venv/bin/python -c "import chessckers_cpp" 2>/dev/null; then
  NATIVE="--native"; echo "leena: native C++ engine -> --native"
else
  echo "leena: chessckers_cpp unavailable -> Python engine (no --native)"
fi
nohup .venv/bin/python -m chessckers_engine.selfplay_workers_only \
  --run-dir "$HOME/chessckers/run" --weights "$HOME/chessckers/run/weights.pt" $NATIVE \
  --workers 4 --worker-id-base 300 --device cpu --sims 400 \
  --d-hidden 256 --c-filters 96 --n-blocks 4 \
  --temperature 1.0 --dirichlet-alpha 0.5 --dirichlet-eps 0.40 \
  --max-plies 200 --weights-poll-seconds 20 --seed 4000 \
  > "$HOME/chessckers/run/workers.log" 2>&1 &
echo $! > "$HOME/chessckers/run/pid"
disown
echo "leena workers launched (pid $(cat "$HOME/chessckers/run/pid"))"
