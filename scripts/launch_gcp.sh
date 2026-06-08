#!/usr/bin/env bash
# GCP self-play CLIENT launcher (Linux). The on-box ExecStart for the chessckers-sp systemd
# unit that scripts/gcp/startup.sh installs. Linux sibling of launch_fleet_leena.sh, stripped
# for a headless GCE box: no caffeinate (servers don't sleep), no en0 bind (the route to the
# trainer's 100.x is over Tailscale's utun). Runs the SAME native cc_selfplay engine as the
# Apple boxes — the NN forward is portable cblas_sgemm, built against OpenBLAS here (startup.sh
# installs libopenblas-dev + runs cpp/build.sh). Reaches the trainer over the tailnet; SERVER is
# injected by the unit. Same client path + shared scripts/fleet.env shape as local/leena, so
# this box CANNOT drift from the rest of the fleet.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/scripts/fleet.env"
: "${SERVER:?set SERVER=http://<trainer-tailnet-ip>:8000}"
ENG="$REPO_ROOT/engine"
PY="$ENG/.venv/bin/python"
RUN="$ENG/weights/run-gcp"            # this box's own client run-dir (mirrors local/leena)
SEED_MIX="$REPO_ROOT/scripts/seed_mix.txt"
WORKERS="${WORKERS:-$(nproc)}"        # one self-play worker per vCPU
# Distinct worker-id-base PER NODE: each worker's RNG is seeded (--seed + worker-id-base + i), so
# two boxes sharing a base self-play byte-identical games (same net + same RNG stream) and the
# second box's data is pure duplicate. Hash the hostname into a wide band clear of local(0)/
# leena(300) so every node — including identical MIG instances — gets its own game stream.
WORKER_ID_BASE="${WORKER_ID_BASE:-$(( 100000 + $(hostname | cksum | cut -d' ' -f1) % 900000 ))}"
cd "$ENG" || exit 1

fleet_export_env
export MACHINE=gcp
# Same seed mix as local/leena (scripts/seed_mix.txt) -> no curriculum drift between boxes.
export CHESSCKERS_START_FEN="$(fleet_seed_fens "$SEED_MIX")"
mkdir -p "$RUN/buffer"
rm -f "$RUN/STOP" 2>/dev/null || true

# Self-update when the server advertises a newer code sha: pull the public origin, rebuild the
# native engine (chessckers_cpp + cc_selfplay against OpenBLAS), then the client re-execs onto
# fresh code (keeps the box wire-aligned with the trainer + its cc_selfplay binary current).
UPDATE_CMD="cd '$REPO_ROOT' && git pull --ff-only && cd '$ENG' && PATH='$ENG/.venv/bin':\$PATH cpp/build.sh"

# fleet_client owns the engine pool: pull net (weights.bin) + params, spawn + supervise N
# cc_selfplay --jobs-local procs, upload games, contribute gate games, self-update (+ native
# rebuild) on a new server version. worker-id-base 400 -> games attribute to [gcp]. The engine
# loads the .bin (self-describing) + reads sims/max-plies/start-fen from the job + env, so the
# arch/device/sims knobs aren't passed here. WORKER_ID_BASE per-node keeps RNG streams distinct.
CC_SELFPLAY="$ENG/cpp/build/cc_selfplay"
[ -x "$CC_SELFPLAY" ] || { echo "cc_selfplay not built at $CC_SELFPLAY (startup.sh runs cpp/build.sh)"; exit 1; }
exec "$PY" -m chessckers_engine.fleet_client \
  --server "$SERVER" --run-dir "$RUN" --client-id "gcp-$(hostname)" --poll-seconds "$FLEET_POLL_S" \
  --update-cmd "$UPDATE_CMD" \
  --queue-depth "$WORKERS" --spawn-engines \
  --engine-binary "$CC_SELFPLAY" \
  --engine-workers "$WORKERS" --engine-worker-id-base "$WORKER_ID_BASE" --engine-seed 4000
