#!/usr/bin/env bash
# Leena HTTP-fleet launcher (lc0-style, client-owns-engine). Starts ONE process: the
# fleet_client, which OWNS the self-play workers. It pulls the net + the canonical
# self-play params, spawns selfplay_workers_only once weights land, restarts them if they
# die, uploads finished games, contributes keep-best GATE games, self-updates on a new
# server code version, and reports worker liveness to the server. Shared shape (arch,
# max-plies, seed mix, sims fallback) comes from scripts/fleet.env so leena CANNOT drift
# from local; only box-specific bits (LAN server, client-id, worker-id-base 300, caffeinate,
# self-update) live here. This is the SAME client path as launch_local_client.sh.
#
# Deploy: from local  `git push leena main`  then on leena
#   cd ~/chessckers && git pull --ff-only && bash scripts/launch_fleet_leena.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/scripts/fleet.env"
SERVER="${SERVER:-http://192.168.68.107:8000}"
ENG="$REPO_ROOT/engine"
PY="$ENG/.venv/bin/python"
RUN="$ENG/weights/run-local"          # leena's own client run-dir (mirrors the local client)
SEED_MIX="$REPO_ROOT/scripts/seed_mix.txt"
cd "$ENG" || exit 1

fleet_export_env
export MACHINE=leena
# Same seed mix as local (scripts/seed_mix.txt) -> no curriculum drift between boxes.
export CHESSCKERS_START_FEN="$(fleet_seed_fens "$SEED_MIX")"

mkdir -p "$RUN/buffer"
rm -f "$RUN/STOP" 2>/dev/null || true

# Keep the Air awake with a STANDALONE detached caffeinate (survives ssh teardown; a
# wrapping one does not). Needs leena on AC power.
pkill -x caffeinate 2>/dev/null || true
nohup caffeinate -ims >/dev/null 2>&1 </dev/null & disown

# Never double-launch. The client owns the workers, so killing it is enough, but also reap
# any stray workers left by an older (separate-launch) script.
pkill -f "chessckers_engine.fleet_client" 2>/dev/null || true
pkill -f "chessckers_engine.selfplay_workers_only" 2>/dev/null || true
sleep 1

# Native C++ engine. ALWAYS rebuild after a code pull: a stale but importable .so silently
# mismatches the Python call surface and crashes the workers at runtime, not import. cmake
# is a uv-pip wheel in the venv bin, so put it on PATH (the venv is never activated here).
# Only pass --native if the rebuild SUCCEEDED — else fall back to the (slower but correct)
# Python engine rather than running a stale ext.
NATIVE=""; BUILD_OK=0
if [ -x cpp/build.sh ]; then
  echo "leena: rebuilding chessckers_cpp (cpp/build.sh)…"
  if PATH="$ENG/.venv/bin:$PATH" cpp/build.sh > "$RUN/cpp_build.log" 2>&1; then BUILD_OK=1
  else echo "leena: native build FAILED (see run-local/cpp_build.log)"; fi
fi
if [ "$BUILD_OK" = 1 ] && "$PY" -c "import chessckers_cpp" 2>/dev/null; then
  NATIVE="--native"; echo "leena: native C++ engine -> --native"
else
  echo "leena: Python engine (no --native) — build failed or ext unavailable"
fi

# Self-update command: when the server advertises a newer code sha than this client booted
# on, pull the bare repo into the tree, rebuild the native ext, and the client re-execs
# itself onto the fresh code (closes the stale-.so failure class). Best-effort (--ff-only);
# if the pull/build fails the box stays on old code and is visibly stale in /status.
UPDATE_CMD="cd '$REPO_ROOT' && git pull --ff-only && cd '$ENG' && PATH='$ENG/.venv/bin':\$PATH cpp/build.sh"

# fleet_client owns the workers: pull net + params, spawn + supervise selfplay_workers_only,
# upload games, contribute gate games, self-update on a new server version. worker-id-base
# 300 -> games attribute to [leena]. --sims is only a FALLBACK for the first-game window
# before the server's selfplay.json is mirrored in; run-local/selfplay.json then governs.
nohup "$PY" -m chessckers_engine.fleet_client \
  --server "$SERVER" --run-dir "$RUN" --client-id leena --poll-seconds $FLEET_POLL_S \
  --bind-interface en0 \
  --update-cmd "$UPDATE_CMD" \
  --spawn-workers -- \
  --workers $FLEET_WORKERS --worker-id-base 300 --seed 4000 \
  --device $FLEET_DEVICE --d-hidden $FLEET_DH --c-filters $FLEET_CF --n-blocks $FLEET_NB \
  --max-plies $FLEET_MAX_PLIES --sims $FLEET_SIMS_FALLBACK --weights-poll-seconds $FLEET_WEIGHTS_POLL_S $NATIVE \
  > "$RUN/fleet_client.log" 2>&1 &
echo $! > "$RUN/client.pid"; disown
echo "leena fleet_client launched (pid $(cat "$RUN/client.pid")) -> $SERVER"
echo "leena up: client owns $FLEET_WORKERS workers (spawned once weights land)."
echo "logs: $RUN/fleet_client.log (client) + $RUN/workers.log (workers)"
