#!/usr/bin/env bash
# Run AZ self-play training with a live spectator UI.
#
# Brings up (idempotently) the scalachess server on :8080 and a static file
# server for chessground/ on :8001, then runs the training loop with
# --watch-dir pointed at chessground/watch/. Open
# http://localhost:8001/spectate.html in a browser to watch.
#
# Any extra args are passed through to the training loop:
#   ./spectate-training.sh --iterations 5 --games-per-iter 5 --sims 25
#
# Override the watch dir with WATCH_DIR=/path/to/dir.
set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
WATCH_DIR=${WATCH_DIR:-$REPO/chessground/watch}
mkdir -p "$WATCH_DIR"

started_pids=()
cleanup() {
    if [ ${#started_pids[@]} -gt 0 ]; then
        for pid in "${started_pids[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
    fi
}
trap cleanup EXIT INT TERM

api_ready() {
    curl -fsS -m 1 -X POST http://localhost:8080/api/game/new \
        -H 'Content-Type: application/json' -d '{}' >/dev/null 2>&1
}
static_ready() {
    curl -fsS -m 1 http://localhost:8001/spectate.html >/dev/null 2>&1
}

# 1) scalachess server
if api_ready; then
    echo "[1/3] scalachess already up on :8080"
else
    echo "[1/3] starting scalachess (sbt run) — this takes ~30s on a cold start…"
    ( cd "$REPO/server" && sbt --error run ) >/tmp/scalachess.log 2>&1 &
    started_pids+=("$!")
    for _ in $(seq 1 120); do
        api_ready && { echo "      ready"; break; }
        sleep 1
    done
    api_ready || { echo "scalachess failed to start; see /tmp/scalachess.log" >&2; exit 1; }
fi

# 2) static file server for chessground/
if static_ready; then
    echo "[2/3] static server already up on :8001"
else
    echo "[2/3] starting static server on :8001…"
    ( cd "$REPO/chessground" && python3 -m http.server 8001 ) >/tmp/spectate-static.log 2>&1 &
    started_pids+=("$!")
    for _ in $(seq 1 10); do
        static_ready && { echo "      ready"; break; }
        sleep 0.5
    done
    static_ready || { echo "static server failed to start; see /tmp/spectate-static.log" >&2; exit 1; }
fi

echo
echo "[3/3] starting training. Open: http://localhost:8001/spectate.html"
echo

cd "$REPO/engine"
uv run python -m chessckers_engine.selfplay_az_loop \
    --watch-dir "$WATCH_DIR" "$@"
