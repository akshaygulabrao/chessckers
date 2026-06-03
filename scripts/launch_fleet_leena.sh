#!/usr/bin/env bash
# Leena HTTP-fleet launcher (lc0-style, client-owns-engine). Starts ONE process: the
# fleet_client, which now OWNS the self-play workers. It pulls the net + the canonical
# self-play params, spawns selfplay_workers_only once weights land, restarts them if
# they die, uploads finished games, contributes keep-best GATE games, self-updates on a
# new server code version, and reports worker liveness to the server — so a dead-worker
# zombie (client up, workers gone) shows up in /status instead of silently producing
# nothing. arch 256/96/4 MUST match the trainer; self-play params (sims/temperature/
# noise) now come LIVE from the server (mirrored into run/selfplay.json), so leena can
# no longer drift from local.
#
# Deploy: from local  `git push leena main`  then on leena
#   cd ~/chessckers && git pull --ff-only && bash scripts/launch_fleet_leena.sh
set -uo pipefail
SERVER="${SERVER:-http://192.168.68.107:8000}"
RUN="$HOME/chessckers/run"
ENG="$HOME/chessckers/engine"
PY="$ENG/.venv/bin/python"
cd "$ENG" || exit 1

export MACHINE=leena CHESSCKERS_MAX_PLIES=200 CHESSCKERS_VALUE_DISCOUNT=0.98 \
       OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# Same single active seed as the local run (scripts/seed_mix.txt: 8-pawn three-king).
export CHESSCKERS_START_FEN='8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1'

mkdir -p "$RUN/buffer"
rm -f "$RUN/STOP" 2>/dev/null || true

# Keep the Air awake with a STANDALONE detached caffeinate (survives ssh teardown;
# a wrapping one does not). Needs leena on AC power.
pkill -x caffeinate 2>/dev/null || true
nohup caffeinate -ims >/dev/null 2>&1 </dev/null & disown

# Never double-launch. The client owns the workers now, so killing it is enough, but
# also reap any stray workers left by an older (separate-launch) script.
pkill -f "chessckers_engine.fleet_client" 2>/dev/null || true
pkill -f "chessckers_engine.selfplay_workers_only" 2>/dev/null || true
sleep 1

# Native C++ engine. ALWAYS rebuild after a code pull: a stale but importable .so
# silently mismatches the Python call surface (e.g. the tree-reuse arg added in
# d2b0834) and crashes the workers at runtime, not import. cmake is a uv-pip wheel in
# the venv bin, so put it on PATH (the venv is never activated here). Only pass
# --native if the rebuild SUCCEEDED — else fall back to the (slower but correct)
# Python engine rather than running a stale ext.
NATIVE=""; BUILD_OK=0
if [ -x cpp/build.sh ]; then
  echo "leena: rebuilding chessckers_cpp (cpp/build.sh)…"
  if PATH="$ENG/.venv/bin:$PATH" cpp/build.sh > "$RUN/cpp_build.log" 2>&1; then BUILD_OK=1
  else echo "leena: native build FAILED (see run/cpp_build.log)"; fi
fi
if [ "$BUILD_OK" = 1 ] && "$PY" -c "import chessckers_cpp" 2>/dev/null; then
  NATIVE="--native"; echo "leena: native C++ engine -> --native"
else
  echo "leena: Python engine (no --native) — build failed or ext unavailable"
fi

# Self-update command: when the server advertises a newer code sha than this client
# booted on, pull the bare repo into the tree, rebuild the native ext, and the client
# re-execs itself onto the fresh code (closes the stale-.so failure class). Best-effort
# (--ff-only); if the pull/build fails the box stays on old code and is visibly stale.
UPDATE_CMD="cd '$HOME/chessckers' && git pull --ff-only && cd '$ENG' && PATH='$ENG/.venv/bin':\$PATH cpp/build.sh"

# fleet_client owns the workers (lc0 client-owns-engine): pull net + params, spawn and
# supervise selfplay_workers_only (its flags follow `--`; the client injects
# --run-dir/--weights and waits for weights before spawning), upload games, contribute
# gate games, and self-update on a new server version. Crash-proof loop. worker-id-base
# 300 -> games attribute to [leena]. --sims here is only a FALLBACK for the first-game
# window before the server's selfplay.json is mirrored in; run/selfplay.json then
# governs sims/temperature/noise live.
nohup "$PY" -m chessckers_engine.fleet_client \
  --server "$SERVER" --run-dir "$RUN" --client-id leena --poll-seconds 15 \
  --update-cmd "$UPDATE_CMD" \
  --spawn-workers -- \
  --workers 4 --worker-id-base 300 --device cpu \
  --d-hidden 256 --c-filters 96 --n-blocks 4 \
  --max-plies 200 --weights-poll-seconds 20 --seed 4000 --sims 100 $NATIVE \
  > "$RUN/fleet_client.log" 2>&1 &
echo $! > "$RUN/client.pid"; disown
echo "leena fleet_client launched (pid $(cat "$RUN/client.pid")) -> $SERVER"
echo "leena up: client owns 4 workers (spawned once weights land)."
echo "logs: run/fleet_client.log (client) + run/workers.log (workers)"
