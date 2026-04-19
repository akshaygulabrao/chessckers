#!/usr/bin/env bash
# Prune every process started by ./start.sh (sbt server + python frontend),
# whether launched in this shell or orphaned from a previous run.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$ROOT/server"
FRONT_DIR="$ROOT/chessground"
FRONT_PORT="${FRONT_PORT:-5173}"
API_PORT="${API_PORT:-8080}"

killed_any=0

# kill_pids <label> <newline-separated-pids>
kill_pids() {
  local label="$1"
  local blob="$2"
  [[ -z "$blob" ]] && return
  local pids
  pids=$(echo "$blob" | tr '\n' ' ' | sed 's/ *$//')
  [[ -z "$pids" ]] && return
  echo "Stopping $label: $pids"
  kill $pids 2>/dev/null || true
  killed_any=1
}

# pids_for_cwd <pgrep-pattern> <cwd-prefix>
pids_for_cwd() {
  local pattern="$1"
  local want="$2"
  local p cwd
  for p in $(pgrep -f "$pattern" 2>/dev/null || true); do
    cwd=$(lsof -a -p "$p" -d cwd -Fn 2>/dev/null | awk '/^n/{sub(/^n/,""); print; exit}')
    if [[ "$cwd" == "$want"* ]]; then
      echo "$p"
    fi
  done
}

# 1. Anything listening on the two ports.
for port in "$API_PORT" "$FRONT_PORT"; do
  blob=$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
  kill_pids "listener on :$port" "$blob"
done

# 2. sbt / forked JVMs launched from the server dir (catches runs not bound to :8080 yet).
blob=$(pids_for_cwd "sbt.*run" "$SERVER_DIR")
kill_pids "sbt in $SERVER_DIR" "$blob"

# 3. python http.server serving the frontend dir.
blob=$(pids_for_cwd "python.* -m http.server" "$FRONT_DIR")
kill_pids "python http.server in $FRONT_DIR" "$blob"

# Give them a moment, then SIGKILL anything still bound to the ports.
sleep 1
for port in "$API_PORT" "$FRONT_PORT"; do
  blob=$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
  if [[ -n "$blob" ]]; then
    pids=$(echo "$blob" | tr '\n' ' ' | sed 's/ *$//')
    echo "Force-killing stragglers on :$port: $pids"
    kill -9 $pids 2>/dev/null || true
    killed_any=1
  fi
done

if [[ $killed_any -eq 0 ]]; then
  echo "Nothing to stop."
else
  echo "Done."
fi
