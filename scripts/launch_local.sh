#!/usr/bin/env bash
# Local TRAINING-SERVER side of the fleet (lc0-style): trainer + arena + fleet_server.
# This box NO LONGER self-plays directly — self-play is a CLIENT now. For loopback local
# self-play run scripts/launch_local_client.sh (exactly the path leena uses over the LAN).
#
#   trainer (MPS)  --run-dir run   random init, drains buffer/, publishes weights.pt + ckpts
#   arena   (CPU)  --run-dir run   seeds best.pt, gates ckpts -> best.pt (keep-best)
#   server  (HTTP) --run-dir run   :PORT — serves gated best.pt + live selfplay.json,
#                                   ingests client games (local + leena) into buffer/
#
# Self-play params live in run/selfplay.json (the ONE source of truth; served at /selfplay,
# mirrored onto every client, applied live per game). Arch comes from scripts/fleet.env so
# trainer / arena / clients can't drift.
#
# Usage:
#   scripts/launch_local.sh            # launch the server side (aborts if a fleet proc is up)
#   FRESH=1 scripts/launch_local.sh    # rm -rf run/ first -> completely new random weights
#   then:  scripts/launch_local_client.sh   # add loopback self-play (its own run-dir)
#
# Tunables (env): SIMS(=fleet fallback) PORT(=8000) LOG(=/tmp/cc_train.log)
#                 PER_GAME_KEEP(=0.5) — per-game downsample frac for the replay window (1.0=off)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/scripts/fleet.env"
ENG="$REPO_ROOT/engine"
PY="$ENG/.venv/bin/python"
RUN="$ENG/weights/run"
LOG="${LOG:-/tmp/cc_train.log}"
SEED_MIX="$REPO_ROOT/scripts/seed_mix.txt"
SIMS="${SIMS:-$FLEET_SIMS_FALLBACK}"; PORT="${PORT:-8000}"
PER_GAME_KEEP="${PER_GAME_KEEP:-0.5}"   # keep ~half of each game's plies in the live window (lc0-style SKIP decorrelation)

say(){ echo "[launch-local] $*" >&2; }

if pgrep -f 'chessckers_engine\.(train_continuous|fleet_server|fleet_arena|fleet_client|selfplay_workers_only)' >/dev/null; then
  say "ABORT: a fleet process is already running. Stop it first:  scripts/stop_local.sh"
  exit 1
fi

if [ -n "${FRESH:-}" ]; then
  say "FRESH: rm -rf $RUN  (completely new random weights)"
  rm -rf "$RUN"
fi
mkdir -p "$RUN/buffer"

say "seed mix: $(grep -cvE '^[[:space:]]*(#|$)' "$SEED_MIX") positions from $SEED_MIX"

# Canonical self-play params -> run/selfplay.json: the ONE source of truth. fleet_server
# serves it at /selfplay; every client (local + leena) mirrors it and applies it live per
# game. Anneal the fleet by editing this file (no relaunch); a client's CLI --sims is only
# the fallback before its first mirror.
cat > "$RUN/selfplay.json" <<JSON
{"sims": $SIMS, "c_puct": 1.5, "temperature": 1.0, "dirichlet_alpha": 0.5, "dirichlet_eps": 0.40, "max_plies": $FLEET_MAX_PLIES}
JSON
say "self-play params -> $RUN/selfplay.json (sims=$SIMS)"

cd "$ENG"

# 1. trainer — random init (no --base), MPS, archive off.
nohup "$PY" -m chessckers_engine.train_continuous \
  --run-dir "$RUN" --no-prime \
  --buffer-cap 300000 --min-buffer 2000 --replay-factor 8 --batch-size 256 \
  --per-game-keep "$PER_GAME_KEEP" \
  --publish-seconds 45 --ckpt-seconds 120 \
  --d-hidden $FLEET_DH --c-filters $FLEET_CF --n-blocks $FLEET_NB --seed 1000 \
  >>"$LOG" 2>&1 &
say "trainer  pid $!  (per-game-keep=$PER_GAME_KEEP)"

# 2. arena — seeds best v0 from weights.pt, then gates each new checkpoint.
nohup "$PY" -m chessckers_engine.fleet_arena \
  --run-dir "$RUN" --seed-mix-file "$SEED_MIX" \
  --d-hidden $FLEET_DH --c-filters $FLEET_CF --n-blocks $FLEET_NB \
  --sims 160 --pairs 4 --threshold 0.55 --side-floor 0.45 \
  --max-plies $FLEET_MAX_PLIES --gate-seconds 60 --device cpu \
  >>"$LOG" 2>&1 &
say "arena    pid $!"

# 3. server — network face of run/: serves gated best.pt + live selfplay.json, ingests
#    client games into buffer/. Local self-play (launch_local_client.sh) and leena both
#    talk to it over HTTP. 0.0.0.0 so leena can reach it on the LAN.
nohup "$PY" -m chessckers_engine.fleet_server \
  --run-dir "$RUN" --host 0.0.0.0 --port "$PORT" \
  >>"$LOG" 2>&1 &
say "server   pid $!  (:$PORT)"

say "up (server side). one log to watch:  tail -f $LOG"
say "add local self-play:  scripts/launch_local_client.sh"
