#!/usr/bin/env bash
# One-shot FRESH fleet redeploy onto the current HEAD commit.
#
#   1. local server : stop the fleet, FRESH relaunch trainer + arena + fleet_server (WIPES
#                     run/, random init) -> the server advertises this commit at
#                     /client-version + serves the live /selfplay params.
#   2. verify       : wait until the local server reports this commit (so step 4 can't race
#                     leena onto a server still on old code).
#   3. local client : FRESH loopback self-play client (own run-local/) against the server.
#   4. leena        : push this commit to leena's bare repo, then ssh in to pull + relaunch
#                     the (client-owns-workers, self-updating) fleet_client.
#
# DESTRUCTIVE: discards run/ AND run-local/ (you chose a fresh restart). Re-runnable.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LEENA="leenagulabrao@Leenas-MacBook-Air.local"
SHA="$(git -C "$REPO" rev-parse --short HEAD)"
LOCAL_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 192.168.68.107)"

echo "[deploy] HEAD=${SHA}  local-ip=${LOCAL_IP}"
echo "[deploy] 1/4 local server: stop + FRESH relaunch (trainer + arena + server; wipes run/)..."
"$REPO/scripts/stop_local.sh" || true
FRESH=1 "$REPO/scripts/launch_local.sh"

echo "[deploy] 2/4 waiting for local server to advertise ${SHA}..."
v=""
for _ in $(seq 1 40); do
  v="$(curl -fsS "http://127.0.0.1:8000/client-version" 2>/dev/null || true)"
  [ "$v" = "$SHA" ] && break
  sleep 1
done
[ "$v" = "$SHA" ] || { echo "[deploy] ABORT: local server on '${v}', expected ${SHA} (check /tmp/cc_train.log)"; exit 1; }
echo "[deploy] local server up on ${SHA}; /selfplay -> $(curl -fsS http://127.0.0.1:8000/selfplay)"

echo "[deploy] 3/4 local client: FRESH loopback self-play (wipes run-local/)..."
FRESH=1 "$REPO/scripts/launch_local_client.sh"

echo "[deploy] 4/4 leena: push ${SHA} + pull + relaunch (SERVER=http://${LOCAL_IP}:8000)..."
git -C "$REPO" push leena main
ssh "$LEENA" "cd ~/chessckers && git pull --ff-only && SERVER=http://${LOCAL_IP}:8000 bash scripts/launch_fleet_leena.sh"

echo "[deploy] done."
echo "  local log : tail -f /tmp/cc_train.log"
echo "  local sp  : tail -f ${REPO}/engine/weights/run-local/workers.log"
echo "  leena log : ssh ${LEENA} tail -f '~/chessckers/engine/weights/run-local/fleet_client.log'"
echo "  fleet     : curl -s http://127.0.0.1:8000/status | python3 -m json.tool"
