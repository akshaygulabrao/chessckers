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

# Native ext is already built on the dev box; pass --native iff it imports (else fall back
# to the slower-but-correct Python engine rather than running a stale ext).
NATIVE=""
if "$PY" -c "import chessckers_cpp" 2>/dev/null; then
  NATIVE="--native"; say "native C++ engine (v1+v2) -> --native"
else
  say "Python engine (no --native) — ext not importable (build: cd engine && cpp/build.sh)"
fi

# fleet_client owns the workers: pull net + live params from the server, spawn + supervise
# selfplay_workers_only, upload finished games, contribute keep-best gate games. No
# --update-cmd (this box is the code source). worker-id-base 0 -> games attribute to [local].
CLIENT_ARGS=(
  -m chessckers_engine.fleet_client
  --server "$SERVER" --run-dir "$RUN" --client-id local --poll-seconds "$FLEET_POLL_S"
  --queue-depth "$WORKERS" --spawn-workers --
  --workers "$WORKERS" --worker-id-base 0 --seed 1000
  --device "$FLEET_DEVICE" --d-hidden "$FLEET_DH" --c-filters "$FLEET_CF" --n-blocks "$FLEET_NB"
  --arch-version "$FLEET_ARCH_VERSION" --tf-blocks "$FLEET_TF_BLOCKS" --tf-heads "$FLEET_TF_HEADS" --tf-ff-mult "$FLEET_TF_FF"
  --max-plies "$FLEET_MAX_PLIES" --sims "$FLEET_SIMS_FALLBACK" --weights-poll-seconds "$FLEET_WEIGHTS_POLL_S"
)
[ -n "$NATIVE" ] && CLIENT_ARGS+=("$NATIVE")

# Foreground: this tab OWNS the local workers. Ctrl-C winds them down (STOP + reap) and exits.
# Loopback (127.0.0.1) is never "local network", so this isn't the macOS TCC fix leena needs
# — here foreground is just for in-tab logs + clean teardown. Streams to the tab AND the log.
cleanup(){ echo; say "stopping local workers…"; touch "$RUN/STOP" 2>/dev/null || true; pkill -f 'chessckers_engine\.selfplay_workers_only' 2>/dev/null || true; }
trap 'cleanup; exit 0' INT TERM
say "local client -> $SERVER  ($WORKERS workers, run-dir $RUN). Ctrl-C stops everything."
say "  worker self-play output also at: $RUN/workers.log"
"$PY" "${CLIENT_ARGS[@]}" 2>&1 | tee "$RUN/fleet_client.log"
cleanup
