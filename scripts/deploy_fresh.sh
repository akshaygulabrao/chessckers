#!/usr/bin/env bash
# One-shot FRESH reset for the manual 3-tab fleet. The fleet now runs as three FOREGROUND
# tabs (launch_server.sh / launch_local.sh / launch_leena.sh), so this script can't occupy
# your tabs — instead it does the DESTRUCTIVE reset + leena push, then prints the three
# commands to paste into three tabs.
#
#   1. stop any running fleet (graceful STOP + sweep).
#   2. WIPE run/ (server) and run-local/ (local client) -> brand-new random weights.
#   3. push HEAD to leena's bare repo + wipe leena's run-local/ (fresh games there too).
#   4. print the 3 tab commands.
#
# DESTRUCTIVE: discards run/ AND run-local/ (local + leena). Re-runnable.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LEENA="${LEENA:-leenagulabrao@Leenas-MacBook-Air.local}"
ENG="$REPO/engine"
SHA="$(git -C "$REPO" rev-parse --short HEAD)"

echo "[deploy] HEAD=${SHA}: stop + WIPE run/ run-local/ (local + leena), push leena."
"$REPO/scripts/stop_local.sh" || true
rm -rf "$ENG/weights/run" "$ENG/weights/run-local"

echo "[deploy] push ${SHA} to leena + wipe leena run-local/…"
git -C "$REPO" push leena main || echo "[deploy] WARN: push to leena failed (leena offline?)"
ssh "$LEENA" "cd ~/chessckers && git pull --ff-only && rm -rf engine/weights/run-local" \
  || echo "[deploy] WARN: leena pull/wipe failed (offline?) — launch_leena.sh will pull when you run it"

cat <<EOF

[deploy] reset done on ${SHA}. Open three terminal tabs and run one each:

  tab 1 (server) :  scripts/launch_server.sh
  tab 2 (local)  :  scripts/launch_local.sh     # once the server tab is serving :8000
  tab 3 (leena)  :  scripts/launch_leena.sh     # once the server tab is serving :8000

Each tab streams its own logs; Ctrl-C in a tab stops that piece.
EOF
