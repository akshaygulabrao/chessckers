#!/usr/bin/env bash
# Run the iter-vs-iter gauntlet. Ensures the scalachess server is up first.
#
#   ./gauntlet.sh --ladder-dir engine/weights/ln --games 30 --sims 50
#   ./gauntlet.sh --challenger PATH --champion PATH --games 30 --sims 50
#
# All extra args pass through to chessckers_engine.gauntlet.
set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
INVOCATION_CWD=$PWD
started_pids=()
cleanup() {
    if [ ${#started_pids[@]} -gt 0 ]; then
        for pid in "${started_pids[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
    fi
}
trap cleanup EXIT INT TERM

# The script cds into engine/ before running, so any path args the user
# typed need to be resolved against the directory they invoked from.
absolutize() {
    case "$1" in
        /*) printf '%s' "$1" ;;
        *)  printf '%s/%s' "$INVOCATION_CWD" "$1" ;;
    esac
}
processed_args=()
prev=""
for arg in "$@"; do
    case "$prev" in
        --ladder-dir|--challenger|--champion)
            processed_args+=("$(absolutize "$arg")") ;;
        *)
            processed_args+=("$arg") ;;
    esac
    prev="$arg"
done

api_ready() {
    curl -fsS -m 1 -X POST http://localhost:8080/api/game/new \
        -H 'Content-Type: application/json' -d '{}' >/dev/null 2>&1
}

if api_ready; then
    echo "[1/2] scalachess already up on :8080"
else
    echo "[1/2] starting scalachess (sbt run) — ~30s on a cold start…"
    ( cd "$REPO/server" && sbt --error run ) >/tmp/scalachess.log 2>&1 &
    started_pids+=("$!")
    for _ in $(seq 1 120); do
        api_ready && { echo "      ready"; break; }
        sleep 1
    done
    api_ready || { echo "scalachess failed to start; see /tmp/scalachess.log" >&2; exit 1; }
fi

echo "[2/2] running gauntlet"
echo
cd "$REPO/engine"
uv run python -m chessckers_engine.gauntlet "${processed_args[@]+"${processed_args[@]}"}"
