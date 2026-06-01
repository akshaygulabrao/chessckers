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
    mt = re.search(r"\[([^\]]*)\]", fen)
    st = "+".join(s.split(":")[0] for s in mt.group(1).split(",")) if mt else "?"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]   # millisecond, matches local
    print(f"{ts}   game #{n} [leena]: {m.get('outcome','?')} in {m.get('plies','?')} plies (seed {st})")
except Exception:
    pass
PYEOF
  done
  [ -f "$WEIGHTS" ] && rsync -az -e "$SSH" "$WEIGHTS" "$LEENA:chessckers/run/weights.pt" 2>/dev/null || true
  sleep 15
done
