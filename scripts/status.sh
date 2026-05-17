#!/usr/bin/env bash
# One-shot dashboard for the local-006 training stack.
#
# Usage:
#   scripts/status.sh                    # uses defaults
#   RUN_DIR=runs/local-007 scripts/status.sh
#
# Probes (in order): local coord, local tmux, sync sidecars, watchdog,
# worker heartbeats, eval log, latest checkpoint, each active remote tmux
# + buffer count (sourced from scripts/active_remotes.env), optional
# vast.ai billing if VAST_INSTANCE is set. One section per topic; no
# command-by-command spelunking required during an incident.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/engine/runs/local-006}"
VAST_INSTANCE="${VAST_INSTANCE:-}"

# Active remotes are sourced from one config; this script doesn't hardcode
# any particular box (so removing/adding a remote is a one-line edit).
ACTIVE_REMOTES=()
[ -f scripts/active_remotes.env ] && source scripts/active_remotes.env

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
dim()  { printf '\033[2m%s\033[0m\n' "$*"; }

bold "=== chessckers training status — $(date '+%Y-%m-%d %H:%M:%S') ==="
bold "RUN_DIR: $RUN_DIR"
echo

bold "── local coord ──"
# Match the python process, not the tmux argv that contains the same string.
# macOS pgrep -af doesn't reliably include argv, so use ps instead.
COORD_PID=$(ps -axo pid,command | grep 'python.*selfplay_az_async --run-dir' \
            | grep -v grep | grep -v 'tmux new-session' | awk '{print $1}' | head -1)
if [ -n "$COORD_PID" ]; then
  ps -p "$COORD_PID" -o pid,etime,command | tail -1 | awk '{print "pid="$1"  etime="$2}'
  # Last 3 trainer/eval log lines.
  tmux capture-pane -t workers -p 2>/dev/null | \
    grep -E "step=|game [0-9]+/[0-9]+|eval @" | tail -3
else
  echo "(no coord process running)"
fi
echo

bold "── local tmux sessions ──"
tmux ls 2>&1 | head -10
echo

bold "── checkpoints + buffer ──"
LATEST_CKPT=$(ls "$RUN_DIR/checkpoints/"*.pt 2>/dev/null | sort -r | head -1)
echo "latest checkpoint:  ${LATEST_CKPT:-(none)}"
if [ -d "$RUN_DIR/buffer" ]; then
  echo "buffer files:       $(ls "$RUN_DIR/buffer" | wc -l | tr -d ' ')"
fi
if [ -f "$RUN_DIR/eval.jsonl" ]; then
  echo "eval cycles:        $(wc -l < "$RUN_DIR/eval.jsonl" | tr -d ' ')"
fi
if [ -f "$RUN_DIR/STOP" ]; then
  echo "STOP file present:  $(stat -f "%Sm" "$RUN_DIR/STOP")"
fi
if [ -f "$RUN_DIR/run_summary.json" ]; then
  echo "run_summary:        $(stat -f "%Sm" "$RUN_DIR/run_summary.json")"
fi
echo

bold "── sync sidecar last activity ──"
for log in /tmp/sync_leena.log /tmp/sync_vast.log; do
  if [ -f "$log" ]; then
    last=$(tail -3 "$log" | tr -d '\n\r' | head -c 200)
    last_mtime=$(stat -f "%Sm" "$log" 2>/dev/null)
    echo "$(basename "$log"):  last write @ $last_mtime"
    [ -n "$last" ] && echo "  $last"
  fi
done
echo

bold "── worker heartbeats ──"
if [ -d "$RUN_DIR/heartbeats" ]; then
  python3 - "$RUN_DIR/heartbeats" <<'PY'
import json, time, os, sys
hb_dir = sys.argv[1]
now = time.time()
rows = []
for name in sorted(os.listdir(hb_dir)):
    if not name.endswith(".json"): continue
    try:
        with open(os.path.join(hb_dir, name)) as f:
            d = json.load(f)
    except Exception:
        continue
    age = now - float(d.get("wall_ts", 0))
    alive = "ALIVE" if age <= 90 else "STALE"
    rows.append((d.get("machine", "?"), int(d.get("worker_id", -1)),
                 d.get("role", "?"), int(d.get("games_played", 0)),
                 age, alive))
rows.sort()
if not rows:
    print("  (none)")
else:
    fmt = "  {:>7}  wid={:<4}  role={:<8}  games={:<6}  age={:>5.0f}s  {}"
    for m, wid, role, g, age, alive in rows:
        print(fmt.format(m, wid, role, g, age, alive))
    total_games_alive = sum(g for _, _, _, g, _, alive in rows if alive == "ALIVE")
    total_games_all = sum(g for _, _, _, g, _, _ in rows)
    print(f"  TOTAL games (alive workers): {total_games_alive}")
    print(f"  TOTAL games (all heartbeats): {total_games_all}")
PY
else
  echo "  (no heartbeats dir yet)"
fi
echo

bold "── watchdog ──"
if [ -f /tmp/watchdog.log ]; then
  echo "log tail:"
  tail -5 /tmp/watchdog.log | sed 's/^/  /'
fi
echo

if [ "${#ACTIVE_REMOTES[@]}" -eq 0 ]; then
  bold "── active remotes ──"
  echo "  (none — local-only run; uncomment lines in scripts/active_remotes.env to add)"
  echo
else
  for entry in "${ACTIVE_REMOTES[@]}"; do
    IFS='|' read -r r_name r_host r_port r_user r_run <<< "$entry"
    bold "── $r_name ($r_user@$r_host:$r_port) ──"
    ssh -o ConnectTimeout=3 -p "$r_port" "$r_user@$r_host" \
      "export PATH=/opt/homebrew/bin:/usr/local/bin:\$PATH
       tmux ls 2>&1 | head -5
       if [ -d '$r_run/buffer' ]; then
         echo \"buffer files: \$(ls '$r_run/buffer' | wc -l | tr -d ' ')\"
       fi
       if [ -d '$r_run/heartbeats' ]; then
         echo \"heartbeat files: \$(ls '$r_run/heartbeats' | wc -l | tr -d ' ')\"
       fi" 2>&1 | sed 's/^/  /' | head -12
    echo
  done
fi

if [ -n "$VAST_INSTANCE" ] && command -v vastai >/dev/null; then
  bold "── vast.ai billing (instance $VAST_INSTANCE) ──"
  vastai show instance "$VAST_INSTANCE" --raw 2>/dev/null | \
    python3 -c "import json,sys; d=json.load(sys.stdin) if sys.stdin.read else None" 2>/dev/null
  vastai show instance "$VAST_INSTANCE" 2>&1 | head -5 | sed 's/^/  /'
  echo
fi

bold "=== end status ==="
