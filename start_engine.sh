#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$ROOT/server"
FRONT_DIR="$ROOT/chessground"
ENGINE_DIR="$ROOT/engine"
FRONT_PORT="${FRONT_PORT:-5173}"
API_PORT="${API_PORT:-8080}"
ENGINE_PORT="${ENGINE_PORT:-8082}"
ENGINE_PLAYER="${ENGINE_PLAYER:-random}"  # 'random' or 'nn'
ENGINE_MODEL="${ENGINE_MODEL:-}"           # optional path to a torch state_dict for ENGINE_PLAYER=nn
FRONT_URL="http://localhost:${FRONT_PORT}/chessckers.html"
API_URL="http://localhost:${API_PORT}"
ENGINE_URL="http://localhost:${ENGINE_PORT}"

pids=()
cleanup() {
  echo
  echo "Shutting down..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

for port in "$API_PORT" "$FRONT_PORT" "$ENGINE_PORT"; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $port is already in use. Stop the existing process first:"
    lsof -nP -iTCP:"$port" -sTCP:LISTEN
    exit 1
  fi
done

echo "[1/3] Starting API server (sbt run) in $SERVER_DIR ..."
( cd "$SERVER_DIR" && exec sbt run ) &
pids+=($!)

echo "[2/3] Serving frontend from $FRONT_DIR on port $FRONT_PORT ..."
( cd "$FRONT_DIR" && exec python3 -m http.server "$FRONT_PORT" >/dev/null 2>&1 ) &
pids+=($!)

echo "Waiting for API at $API_URL ..."
for _ in $(seq 1 60); do
  if curl -fsS -o /dev/null --max-time 1 -X POST "$API_URL/api/game/new" \
      -H 'Content-Type: application/json' -d '{}' 2>/dev/null; then
    echo "API up."
    break
  fi
  sleep 1
done

echo "[3/3] Starting engine on port $ENGINE_PORT (player=$ENGINE_PLAYER) ..."
( cd "$ENGINE_DIR" \
  && API_URL="$API_URL" \
     ENGINE_PORT="$ENGINE_PORT" \
     ENGINE_PLAYER="$ENGINE_PLAYER" \
     ENGINE_MODEL="$ENGINE_MODEL" \
     exec uv run python -m chessckers_engine ) &
pids+=($!)

if command -v open >/dev/null 2>&1; then
  open "$FRONT_URL"
else
  echo "Open $FRONT_URL in your browser."
fi

echo
echo "Frontend: $FRONT_URL"
echo "API:      $API_URL"
echo "Engine:   $ENGINE_URL ($ENGINE_PLAYER opponent)"
echo "Press Ctrl+C to stop all."
wait
