#!/usr/bin/env bash
# Stream leena's self-play games into the trainer's progress log AND feed them
# to training. Every 15s: MOVE leena's new game pkls (+ .meta) into the trainer's
# ingest dir (--remove-source-files = each game pulled exactly once), log one
# timestamped line per new game using the SHARED game counter (/tmp/cc_gamecount,
# flock'd — same one the trainer bumps, so `game #N` is one sequence across
# machines), then push current weights up to leena. O_APPEND on the shared log
# is safe alongside the append-mode trainer.
set -uo pipefail
LEENA=leenagulabrao@192.168.68.183
ENG=/Users/ox/AAworkspace/chessckers/engine
INGEST="$ENG/weights/run/buffer"
WEIGHTS="$ENG/weights/run/weights.pt"
LOG=/tmp/cc_train.log
PY="$ENG/.venv/bin/python"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
PAUSE="$ENG/weights/run/PAUSE_LEENA"        # trainer touches this across its train+save phase
paused=0; pause_started=0; MAX_PAUSE=600    # SIGSTOP leena while present; force-resume after MAX_PAUSE (stale-marker guard)
mkdir -p "$INGEST"
while true; do
  out=$(rsync -ai --remove-source-files -e "$SSH" "$LEENA:chessckers/run/buffer/" "$INGEST/" 2>/dev/null || true)
  echo "$out" | awk '/\.pkl\.meta$/{print $NF}' | while read -r meta; do
    f="$INGEST/$meta"; [ -f "$f" ] || continue
    "$PY" - "$f" <<'PYEOF' >> "$LOG" 2>/dev/null
import json, sys, re, fcntl
from datetime import datetime
COUNTER = "/tmp/cc_gamecount"
try:
    m = json.load(open(sys.argv[1]))
    with open(COUNTER, "a+") as cf:           # shared atomic counter (trainer + sync)
        fcntl.flock(cf, fcntl.LOCK_EX)
        cf.seek(0); cur = cf.read().strip()
        n = (int(cur) if cur.isdigit() else 0) + 1
        cf.seek(0); cf.truncate(); cf.write(str(n))
        fcntl.flock(cf, fcntl.LOCK_UN)
    fen = m.get("seed_fen") or ""
    board = fen.split(" ")[0]
    n_wp = board.split("[")[0].count("P")            # White pawns (match _seed_tag)
    mt = re.search(r"\[([^\]]*)\]", fen)
    st = "+".join(s.split(":")[0] for s in mt.group(1).split(",")) if mt else "?"
    if mt and n_wp:
        st = f"{st}+{n_wp}P"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]   # millisecond, matches local
    print(f"{ts}   game #{n} [leena]: {m.get('outcome','?')} in {m.get('plies','?')} plies (seed {st})")
except Exception:
    pass
PYEOF
  done
  [ -f "$WEIGHTS" ] && rsync -az -e "$SSH" "$WEIGHTS" "$LEENA:chessckers/run/weights.pt" 2>/dev/null || true
  # Pause leena across the trainer's train+save phase (PAUSE_LEENA marker) so it
  # resumes on the fresh weights just pushed above — SIGSTOP/SIGCONT the workers,
  # no leena-side code. Safety: force-resume if the marker goes stale (trainer died).
  now=$(date +%s)
  if [ -f "$PAUSE" ]; then
    if [ "$paused" = "0" ]; then
      $SSH "$LEENA" 'pkill -STOP -f selfplay_workers_only; pkill -STOP -f multiprocessing.spawn' 2>/dev/null || true
      paused=1; pause_started=$now
      echo "$(date '+%Y-%m-%d %H:%M:%S,000')   [sync] leena paused (trainer training)" >> "$LOG"
    elif [ $((now - pause_started)) -gt "$MAX_PAUSE" ]; then
      $SSH "$LEENA" 'pkill -CONT -f selfplay_workers_only; pkill -CONT -f multiprocessing.spawn' 2>/dev/null || true
      paused=0
      echo "$(date '+%Y-%m-%d %H:%M:%S,000')   [sync] leena force-resumed (pause > ${MAX_PAUSE}s — stale marker?)" >> "$LOG"
    fi
  elif [ "$paused" = "1" ]; then
    $SSH "$LEENA" 'pkill -CONT -f selfplay_workers_only; pkill -CONT -f multiprocessing.spawn' 2>/dev/null || true
    paused=0
    echo "$(date '+%Y-%m-%d %H:%M:%S,000')   [sync] leena resumed (fresh weights)" >> "$LOG"
  fi
  sleep 15
done
