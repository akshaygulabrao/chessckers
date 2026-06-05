#!/usr/bin/env bash
# LEENA self-play CLIENT, launched FROM this (the server) box in its own foreground tab.
# Thin `ssh -t` wrapper: push HEAD to leena, pull it there, then run the on-leena launcher in
# FOREGROUND mode so leena's fleet_client stays a child of THIS ssh session.
#
# Why foreground matters for leena specifically: an ssh-orphaned daemon (nohup/&/disown) is
# denied LAN access by macOS "Local Network" privacy (TCC) and can't reach the server on
# 192.168.x; a process attached to a live, granted session reaches the LAN fine. So keep this
# tab open — closing it (or Ctrl-C) stops leena's client + workers + its caffeinate cleanly.
#
# Server already up?  Run scripts/launch_server.sh in another tab first.
#
# Usage (in its own tab):
#   scripts/launch_leena.sh
#   NO_PUSH=1 scripts/launch_leena.sh    # skip the git push (just restart leena on its code)
#   SERVER=http://10.0.0.5:8000 scripts/launch_leena.sh   # override the server URL
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LEENA="${LEENA:-leenagulabrao@Leenas-MacBook-Air.local}"
# The server URL leena should hit = THIS box's LAN IP (en0, en1 fallback). Override via SERVER.
LOCAL_IP="${LOCAL_IP:-$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 192.168.68.107)}"
SERVER="${SERVER:-http://${LOCAL_IP}:8000}"

# Push HEAD to leena's bare repo so the pull below lands current code (best-effort: if leena
# is unreachable the ssh will fail anyway with a clear error).
if [ -z "${NO_PUSH:-}" ]; then
  echo "[launch-leena] push $(git -C "$REPO_ROOT" rev-parse --short HEAD) -> leena"
  git -C "$REPO_ROOT" push leena main || echo "[launch-leena] WARN: push failed (continuing; leena will pull what it can)"
fi

echo "[launch-leena] ssh $LEENA  (server $SERVER, FOREGROUND). Keep this tab open; Ctrl-C stops leena."
exec ssh -t "$LEENA" \
  "cd ~/chessckers && git pull --ff-only && SERVER='$SERVER' FOREGROUND=1 bash scripts/launch_fleet_leena.sh"
