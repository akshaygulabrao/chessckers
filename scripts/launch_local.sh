#!/usr/bin/env bash
# LOCAL self-play CLIENT (lc0-style client-owns-engine), over loopback to the local
# fleet_server. Runs in the FOREGROUND, owning this terminal tab — the client + its workers
# stream here and Ctrl-C winds them down. The SAME path leena uses (launch_leena.sh ->
# launch_fleet_leena.sh), minus self-update, since this box IS the code source. It uses its
# OWN run-dir so its buffer/weights don't collide with the trainer's run/: the workers write
# games into run-local/buffer, the client uploads them over HTTP to the server, which lands
# them in the trainer's run/buffer.
#
# Run the SERVER side first (other tab):  scripts/launch_server.sh
#
# Usage (in its own tab):
#   scripts/launch_local.sh           # start loopback self-play
#   FRESH=1 scripts/launch_local.sh   # rm -rf run-local/ first (drop stale games)
#
# Tunables (env): SERVER(=http://127.0.0.1:8000) WORKERS(=fleet default)
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/scripts/fleet.env"
ENG="$REPO_ROOT/engine"
PY="$ENG/.venv/bin/python"
SERVER="${SERVER:-http://127.0.0.1:8000}"
RUN="$ENG/weights/run-local"          # the CLIENT's own run-dir (NOT the trainer's run/)
SEED_MIX="$REPO_ROOT/scripts/seed_mix.txt"
WORKERS="${WORKERS:-$FLEET_WORKERS}"

say(){ echo "[launch-local] $*" >&2; }
cd "$ENG"

fleet_export_env
export MACHINE=local
export CHESSCKERS_START_FEN="$(fleet_seed_fens "$SEED_MIX")"

if [ -n "${FRESH:-}" ]; then
  say "FRESH: rm -rf $RUN  (drop any stale client games)"
  rm -rf "$RUN"
fi
mkdir -p "$RUN/buffer"
rm -f "$RUN/STOP" 2>/dev/null || true

# Never double-launch the local client (it owns the workers); reap any stray local workers.
pkill -f 'chessckers_engine\.fleet_client .*run-local' 2>/dev/null || true
sleep 1

# lc0-split cutover (Phase 3B-3): the engine is the native cc_selfplay binary, not the
# Python worker. Require it — there is no Python self-play fallback anymore.
CC_SELFPLAY="$ENG/cpp/build/cc_selfplay"
if [ ! -x "$CC_SELFPLAY" ]; then
  say "cc_selfplay not built at $CC_SELFPLAY — build it: cd engine && cpp/build.sh"; exit 1
fi
say "native C++ engine -> cc_selfplay --jobs-local ($WORKERS procs)"

# fleet_client owns the engine pool: pull net (weights.bin) + live params from the server,
# spawn + supervise N cc_selfplay --jobs-local procs, upload finished games, contribute
# keep-best gate games. No --update-cmd (this box is the code source). worker-id-base 0 ->
# games attribute to [local]. cc_selfplay loads the .bin (self-describing) + reads sims/
# max-plies/start-fen from the job + env, so the arch/device/sims knobs aren't passed here.
CLIENT_ARGS=(
  -m chessckers_engine.fleet_client
  --server "$SERVER" --run-dir "$RUN" --client-id local --poll-seconds "$FLEET_POLL_S"
  --queue-depth "$WORKERS" --spawn-engines
  --engine-binary "$CC_SELFPLAY"
  --engine-workers "$WORKERS" --engine-worker-id-base 0 --engine-seed 1000
)

# Foreground: this tab OWNS the local engines. Ctrl-C winds them down (STOP + reap) and exits.
# Loopback (127.0.0.1) is never "local network", so this isn't the macOS TCC fix leena needs
# — here foreground is just for in-tab logs + clean teardown. Streams to the tab AND the log.
cleanup(){ echo; say "stopping local engines…"; touch "$RUN/STOP" 2>/dev/null || true; pkill -f 'cc_selfplay .*--jobs-local' 2>/dev/null || true; }
trap 'cleanup; exit 0' INT TERM
say "local client -> $SERVER  ($WORKERS engines, run-dir $RUN). Ctrl-C stops everything."
say "  engine self-play output also at: $RUN/engine-*.log"
"$PY" "${CLIENT_ARGS[@]}" 2>&1 | tee "$RUN/fleet_client.log"
cleanup
