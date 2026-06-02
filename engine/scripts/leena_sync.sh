#!/usr/bin/env bash
# Stream leena's self-play games into the trainer's progress log AND feed them
# to training. Every 15s: MOVE leena's new game pkls (+ .meta) into the trainer's
# ingest dir (--remove-source-files = each game pulled exactly once), log one
# timestamped line per new game using the SHARED game counter (/tmp/cc_gamecount,
# flock'd — same one the trainer bumps, so `game #N` is one sequence across
# machines), then push current weights up to leena. O_APPEND on the shared log
# is safe alongside the append-mode trainer.
set -uo pipefail
LEENA=leenagulabrao@Leenas-MacBook-Air.local   # Bonjour hostname — survives leena's DHCP IP changes
ENG=/Users/ox/AAworkspace/chessckers/engine
INGEST="$ENG/weights/run/buffer"
WEIGHTS="$ENG/weights/run/weights.pt"
LOG=/tmp/cc_train.log
PY="$ENG/.venv/bin/python"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
PAUSE="$ENG/weights/run/PAUSE_LEENA"        # trainer touches this across its train+save phase
paused=0; pause_started=0; MAX_PAUSE=600    # SIGSTOP leena while present; force-resume after MAX_PAUSE (stale-marker guard)
last_w_mtime=0                              # only push (+log) weights to leena when train_continuous republishes a newer file
STOP="$ENG/weights/run/STOP"                # trainer touches this at the game target -> tear down leena + exit
mkdir -p "$INGEST"
while true; do
  # Run ended? train_continuous touches STOP at the game target -> stop leena + exit.
  if [ -f "$STOP" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S,000')   [sync] run STOP detected -> stopping leena + exiting sync" >> "$LOG"
    $SSH "$LEENA" 'pkill -f selfplay_workers_only; pkill -f multiprocessing.spawn' 2>/dev/null || true
    break
  fi
  # Pull leena's new games into the shared buffer; train_continuous drains + LOGS
  # them (local + leena alike), so there is no per-game logging here anymore.
  rsync -ai --remove-source-files -e "$SSH" "$LEENA:chessckers/run/buffer/" "$INGEST/" >/dev/null 2>&1 || true
  # Push weights to leena ONLY when train_continuous republished a newer file
  # (mtime-gated) -> no redundant re-push each cycle; logs the leena weights-fetch.
  if [ -f "$WEIGHTS" ]; then
    w_mtime=$(stat -f %m "$WEIGHTS" 2>/dev/null || echo 0)
    if [ "$w_mtime" != "$last_w_mtime" ] && rsync -az -e "$SSH" "$WEIGHTS" "$LEENA:chessckers/run/weights.pt" 2>/dev/null; then
      echo "$(date '+%Y-%m-%d %H:%M:%S,000')   [sync] pushed fresh weights -> leena" >> "$LOG"
      last_w_mtime="$w_mtime"
    fi
  fi
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
