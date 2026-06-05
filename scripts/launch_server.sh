#!/usr/bin/env bash
# SERVER side of the fleet (lc0-style): trainer + arena + fleet_server. Runs in the
# FOREGROUND, owning this terminal tab — the three procs stream their (labeled) logs here and
# Ctrl-C stops all three. This box does NOT self-play; self-play is a CLIENT
# (scripts/launch_local.sh loopback, scripts/launch_leena.sh over the LAN), each in its own
# tab. Open this tab first.
#
#   trainer (MPS)  --run-dir run   random init, drains buffer/, publishes weights.pt + ckpts
#   arena   (tally) --run-dir run   seeds best.pt, gates ckpts -> best.pt (keep-best; plays no game)
#   server  (HTTP) --run-dir run   :PORT — serves gated best.pt + live selfplay.json,
#                                   ingests client games (local + leena) into buffer/
#
# Self-play params live in run/selfplay.json (the ONE source of truth; served at /selfplay,
# mirrored onto every client, applied live per game). Arch comes from scripts/fleet.env so
# trainer / arena / clients can't drift.
#
# Usage (in its own tab):
#   scripts/launch_server.sh            # launch the server side (aborts if a fleet proc is up)
#   FRESH=1 scripts/launch_server.sh    # rm -rf run/ first -> completely new random weights
#   then, in two more tabs:  scripts/launch_local.sh   and   scripts/launch_leena.sh
#
# Tunables (env): SIMS(=fleet fallback) PORT(=8000)
#                 PER_GAME_KEEP(=0.5) — per-game downsample frac for the replay window (1.0=off)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/scripts/fleet.env"
ENG="$REPO_ROOT/engine"
PY="$ENG/.venv/bin/python"
RUN="$ENG/weights/run"
SEED_MIX="$REPO_ROOT/scripts/seed_mix.txt"
SIMS="${SIMS:-$FLEET_SIMS_FALLBACK}"; PORT="${PORT:-8000}"
PER_GAME_KEEP="${PER_GAME_KEEP:-0.5}"   # keep ~half of each game's plies in the live window (lc0-style SKIP decorrelation)

say(){ echo "[launch-server] $*" >&2; }

if pgrep -f 'chessckers_engine\.(train_continuous|fleet_server|fleet_arena|fleet_client|selfplay_workers_only)' >/dev/null; then
  say "ABORT: a fleet process is already running. Stop it first:  scripts/stop_local.sh"
  exit 1
fi

if [ -n "${FRESH:-}" ]; then
  say "FRESH: rm -rf $RUN  (completely new random weights)"
  rm -rf "$RUN"
fi
mkdir -p "$RUN/buffer"
rm -f "$RUN/STOP" 2>/dev/null || true   # clear a stale STOP (interrupted stop_local.sh), else the arena exits at startup

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

# Per-proc logs under run/ so the tab can stream all three, labeled by tail -F. (No detached
# nohup: this tab OWNS the server side; Ctrl-C stops everything.)
TLOG="$RUN/trainer.log"; ALOG="$RUN/arena.log"; SLOG="$RUN/server.log"
: > "$TLOG"; : > "$ALOG"; : > "$SLOG"
pids=()

# 1. trainer — random init (no --base), MPS, archive off.
"$PY" -m chessckers_engine.train_continuous \
  --run-dir "$RUN" --no-prime \
  --buffer-cap 300000 --min-buffer 2000 --replay-factor 8 --batch-size 256 \
  --per-game-keep "$PER_GAME_KEEP" \
  --publish-seconds 45 --ckpt-seconds 1800 \
  --d-hidden $FLEET_DH --c-filters $FLEET_CF --n-blocks $FLEET_NB --seed 1000 \
  >"$TLOG" 2>&1 &
pids+=($!); say "trainer  pid $!  -> $TLOG  (per-game-keep=$PER_GAME_KEEP)"

# 2. arena — seeds best v0 from weights.pt, then gates each new checkpoint.
"$PY" -m chessckers_engine.fleet_arena \
  --run-dir "$RUN" --seed-mix-file "$SEED_MIX" \
  --d-hidden $FLEET_DH --c-filters $FLEET_CF --n-blocks $FLEET_NB \
  --sims 160 --pairs 4 --threshold 0.55 \
  --ladder-rungs all --no-regress 0.50 \
  --max-plies $FLEET_MAX_PLIES --gate-seconds 60 \
  >"$ALOG" 2>&1 &
pids+=($!); say "arena    pid $!  -> $ALOG"

# 3. server — network face of run/: serves gated best.pt + live selfplay.json, ingests
#    client games into buffer/. Local self-play (launch_local.sh) and leena both talk to it
#    over HTTP. 0.0.0.0 so leena can reach it on the LAN.
"$PY" -m chessckers_engine.fleet_server \
  --run-dir "$RUN" --host 0.0.0.0 --port "$PORT" \
  >"$SLOG" 2>&1 &
pids+=($!); say "server   pid $!  -> $SLOG  (:$PORT)"

# Foreground: kill all three on Ctrl-C (and on any exit), then stream their logs into this
# tab until then. tail -F labels each section so trainer/arena/server stay distinguishable.
cleanup(){ [ "${#pids[@]}" -gt 0 ] && kill "${pids[@]}" 2>/dev/null || true; }
trap 'echo; say "stopping trainer+arena+server…"; cleanup; exit 0' INT TERM
trap cleanup EXIT
say "up (server side) on :$PORT. Ctrl-C stops all three. Streaming logs:"
tail -n +1 -F "$TLOG" "$ALOG" "$SLOG"
