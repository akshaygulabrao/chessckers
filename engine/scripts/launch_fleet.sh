#!/usr/bin/env bash
# lc0-style distributed self-play — CLIENT + SERVER + TRAINER.
#
#   TRAINER  (train_continuous)  ─ shares SERVER_RUN with the server (FS)
#   SERVER   (fleet_server)      ─ HTTP face of SERVER_RUN: distributes the net,
#                                  ingests games into SERVER_RUN/buffer
#   CLIENT   (fleet_client)      ─ pulls the net / pushes games over HTTP; the
#                                  self-play workers it fronts are just
#                                  selfplay_workers_only. The LOCAL machine is a
#                                  client too (over loopback) — same code path
#                                  leena / volunteers use, so it's validated here.
#
# This replaces the rsync leena_sync sidecar with a real network server. Leena /
# volunteers join later by running fleet_client against http://<this-host>:PORT.
#
# Stop the whole run: touch "$SERVER_RUN/STOP"  (trainer + server + clients tear down)
set -uo pipefail
ENG=/Users/ox/AAworkspace/chessckers/engine
PY="$ENG/.venv/bin/python"
cd "$ENG"

PORT="${PORT:-8000}"
WORKERS="${WORKERS:-4}"
SIMS="${SIMS:-400}"
MAX_PLIES="${MAX_PLIES:-200}"
BASE="${BASE:-}"                       # empty = cold random init (matches the prior run)
SERVER_RUN="$ENG/weights/run"          # trainer + server share this
LOCAL_RUN="$ENG/weights/run_local"     # local self-play client's own run-dir
LOG=/tmp/cc_train.log                  # unified trainer dashboard (per-game lines)
LAN_IP=$(ipconfig getifaddr en0 || ipconfig getifaddr en1 || echo 127.0.0.1)

# Arch — MUST match across trainer + workers (native .bin export is arch-sensitive).
DH=256; CF=96; NB=4
# Curriculum: the SAME canonical seed mix leena_launch.sh reads, so local + leena
# self-play the identical distribution into the shared buffer.
SEED_MIX_FILE="$ENG/scripts/seed_mix.txt"
MIX=$(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$SEED_MIX_FILE" | paste -sd ';' -)
[ -n "$MIX" ] || { echo "empty/missing seed mix at $SEED_MIX_FILE" >&2; exit 1; }

echo "=== launch_fleet: server :$PORT  LAN=$LAN_IP  workers=$WORKERS sims=$SIMS ==="

# --- clean prior run state (both run-dirs) -------------------------------------
for rd in "$SERVER_RUN" "$LOCAL_RUN"; do
  mkdir -p "$rd/buffer"
  rm -f "$rd"/STOP "$rd"/weights.pt "$rd"/iter-async-*.pt "$rd"/*.json \
        "$rd"/exit_code "$rd"/native_*.bin "$rd"/buffer/* 2>/dev/null || true
done

base_arg=(); [ -n "$BASE" ] && base_arg=(--base "$BASE")

# --- 1. TRAINER (publishes weights.pt to SERVER_RUN immediately) ---------------
MACHINE=local nohup "$PY" -m chessckers_engine.train_continuous \
  --run-dir "$SERVER_RUN" ${base_arg[@]+"${base_arg[@]}"} --no-prime \
  --buffer-cap 300000 --min-buffer 2000 --replay-factor 8 --batch-size 256 \
  --publish-seconds 45 --ckpt-seconds 120 \
  --d-hidden $DH --c-filters $CF --n-blocks $NB --seed 1000 \
  >> "$LOG" 2>&1 & disown
echo "trainer pid=$! (-> $LOG)"

# --- 2. SERVER -----------------------------------------------------------------
nohup "$PY" -m chessckers_engine.fleet_server \
  --run-dir "$SERVER_RUN" --host 0.0.0.0 --port "$PORT" \
  >> /tmp/cc_fleet_server.log 2>&1 & disown
echo "server  pid=$! (-> /tmp/cc_fleet_server.log)"

# wait for the server to answer + have an initial net to hand out
for i in $(seq 1 30); do
  v=$("$PY" - "$PORT" <<'PY' 2>/dev/null || true
import sys,urllib.request
try: print(urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/version",timeout=2).read().decode())
except Exception: print("none")
PY
)
  [ "$v" != "none" ] && [ -n "$v" ] && { echo "server net version: $v"; break; }
  sleep 1
done

# --- 3. LOCAL CLIENT (loopback) — pulls net -> LOCAL_RUN/weights.pt -------------
MACHINE=local nohup "$PY" -m chessckers_engine.fleet_client \
  --server "http://127.0.0.1:$PORT" --run-dir "$LOCAL_RUN" --poll-seconds 15 \
  >> /tmp/cc_fleet_client_local.log 2>&1 & disown
echo "client  pid=$! (-> /tmp/cc_fleet_client_local.log)"

# workers exit immediately if weights.pt is absent at launch — wait for the
# client's first pull to land it.
echo -n "waiting for client to pull weights.pt"
for i in $(seq 1 30); do
  [ -f "$LOCAL_RUN/weights.pt" ] && { echo " ... ok"; break; }
  echo -n "."; sleep 1
done

# --- 4. LOCAL WORKERS (self-play; share LOCAL_RUN with the client) -------------
MACHINE=local CHESSCKERS_START_FEN="$MIX" CHESSCKERS_MAX_PLIES=$MAX_PLIES \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
nohup "$PY" -m chessckers_engine.selfplay_workers_only --native \
  --run-dir "$LOCAL_RUN" --weights "$LOCAL_RUN/weights.pt" \
  --workers "$WORKERS" --worker-id-base 0 --device cpu --sims "$SIMS" \
  --d-hidden $DH --c-filters $CF --n-blocks $NB \
  --temperature 1.0 --dirichlet-alpha 0.5 --dirichlet-eps 0.40 \
  --max-plies "$MAX_PLIES" --weights-poll-seconds 20 --seed 1000 \
  >> /tmp/cc_workers_local.log 2>&1 & disown
echo "workers pid=$! (-> /tmp/cc_workers_local.log)"

echo
echo "=== fleet up. clients join with: ==="
echo "  python -m chessckers_engine.fleet_client --server http://$LAN_IP:$PORT --run-dir <run> --poll-seconds 15"
echo "watch:  tail -f $LOG   |   curl -s http://127.0.0.1:$PORT/status"
echo "stop:   touch $SERVER_RUN/STOP"
