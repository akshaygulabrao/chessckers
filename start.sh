#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$ROOT/server"
FRONT_DIR="$ROOT/chessground"
FRONT_PORT="${FRONT_PORT:-5173}"
FRONT_URL="http://localhost:${FRONT_PORT}/chessckers.html"

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

echo "Starting API server (sbt run) in $SERVER_DIR ..."
( cd "$SERVER_DIR" && sbt run ) &
pids+=($!)

echo "Serving frontend from $FRONT_DIR on port $FRONT_PORT ..."
( cd "$FRONT_DIR" && python3 -m http.server "$FRONT_PORT" ) &
pids+=($!)

sleep 2
if command -v open >/dev/null 2>&1; then
  open "$FRONT_URL"
else
  echo "Open $FRONT_URL in your browser."
fi

echo
echo "Frontend: $FRONT_URL"
echo "API:      http://localhost:8080"
echo "Press Ctrl+C to stop both."
wait
