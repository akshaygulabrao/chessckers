#!/usr/bin/env bash
# bench_resume.sh — self-healing cron driver for mate_bench experiments.
# 2026-07-24: rewritten single-arm for RUN 26 — Gumbel S2 (Sequential Halving,
# --visits=64 --gumbel-sh --gumbel-m=16), TRIALS=2, trainer seeds 0..TRIALS-1.
# The control arm is run 25's gates-off arm A (already banked in
# BENCH_RESULTS.jsonl + bench_trials tars) and is NOT re-run.
# Metric: search-visits-to-crossing (bench_visits.py, ops-noise-immune).
#
# Why cron: the 07-20 first attempt drove the two-arm chain from inside tmux;
# the box's cgroup OOM killer (memory.events at the time: oom 202 /
# oom_kill 117 — also the prime suspect for the run-23/24 trainer SIGKILLs)
# took out the tmux server between trials, killing the driver together with
# the fleet. Cron re-fires every 5 min under flock; the experiment state is
# reconstructed from BENCH_RESULTS.jsonl (per-arm trial-stamp counts), and
# every step is resumable because mate_bench crossings are retro-exact from
# training_games.created_at.
#
# Install on the box:
#   */5 * * * * flock -n /workspace/chessckers/bench_resume.lock bash /workspace/chessckers/bench_resume.sh >> /workspace/chessckers/bench_watch.log 2>&1
# Disarm: remove that cron line (and pkill -f 'mate_benc[h].py --trials').
#
# LOCK-FD FOOTGUN (bit us 07-20 20:25): cron's flock holds the lock on fd 3,
# inherited by every child. `tmux new-session` DAEMONIZES a tmux server that
# keeps fd 3 open forever -> the lock stays held after this script exits and
# all later fires silently no-op. Every child that can spawn a daemon
# (restart_fleet, mate_bench -> relaunch_trial -> restart_fleet) MUST run with
# 3>&- . If the wedge ever recurs: rm the lock file (relocks on a fresh inode).
set -uo pipefail
export PATH=/usr/local/go/bin:/usr/bin:/usr/local/bin:/usr/sbin:/bin:$PATH
WS=/workspace/chessckers
ENG=$WS/engine
SRV=$WS/lczero-server
DB=$SRV/chessckers.db
RES=$WS/BENCH_RESULTS.jsonl
A_NAME=run27_e8d8_puct64_bench
TRIALS=2
# RUN 27 = the run-26 ABLATION. Same 64-visit budget, but the run-25 control's
# ROOT ALGORITHM (PUCT + Dirichlet + temperature) instead of Gumbel+SH — i.e.
# GUMBEL_SH is deliberately NOT set, so bootstrap emits
# ["--noise-epsilon=0.25","--noise-alpha=0.3","--temperature=1.0","--tempdecay-moves=15","--visits=64"].
# Completes the 2x2 with run-25 arm A (800v PUCT) and run-26 (64v Gumbel+SH):
#   800v PUCT vs 64v PUCT  -> the budget effect
#   64v PUCT  vs 64v SH    -> the algorithm effect (the open question)
BASE_ENV="ARCH_VERSION=v5 C_FILTERS=64 N_BLOCKS=6 SE_RATIO=8 POLICY_TARGET=improved VALUE_Q_RATIO=0 EMA_DECAY=0.99 PUBLISH_GAMES=400 PARALLELISM=32 VISITS=64"
log(){ echo "[resume $(date -u '+%m-%d %H:%M')] $*"; }

# Liveness heartbeat BEFORE any early exit: the 07-20 flock wedge was invisible
# because a blocked fire exits silently — `stat` this file to see the cron is alive.
date -u '+%F %T' > "$WS/bench_resume.heartbeat" 2>/dev/null || true

# A watcher already driving? Nothing to do.
pgrep -f 'mate_benc[h].py --trials' >/dev/null && exit 0

read -r A_DONE < <(python3 - "$RES" "$A_NAME" <<'PYEOF'
import json, sys
path, an = sys.argv[1:3]
a = 0
try:
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not r.get("trial"):
            continue
        if r.get("run_name") == an:
            a += 1
except FileNotFoundError:
    pass
print(a)
PYEOF
)
if [ "$A_DONE" -ge "$TRIALS" ]; then
  exit 0   # experiment complete; leaving the cron line is a cheap no-op
fi
NAME=$A_NAME; DONE=$A_DONE
ENV_LINE="RUN_NAME=$A_NAME $BASE_ENV SEED=$DONE"
REMAINING=$((TRIALS - DONE))
log "driving $NAME: done=$DONE remaining=$REMAINING (seed base $DONE)"

# Keep the @reboot cron line in sync with the arm + next seed — it is the
# config mate_bench's between-trial relaunches (and a real reboot) read.
( crontab -l 2>/dev/null | grep -v restart_fleet.sh; \
  echo "@reboot $ENV_LINE $SRV/scripts/restart_fleet.sh --boot >> /workspace/restart_fleet.log 2>&1" ) | crontab -

# Fleet state: fresh launch if the DB is missing or belongs to the other arm
# (arm switch, or death mid-reset); warm resume if the right run is on disk but
# the fleet is down (OOM mid-trial — run clock survives, honest wall-clock);
# leave a healthy fleet alone.
db_run(){
  python3 -c "
import sqlite3
try:
    c = sqlite3.connect('file:$DB?mode=ro', uri=True, timeout=5)
    r = c.execute('SELECT description FROM training_runs ORDER BY id DESC LIMIT 1').fetchone()
    print(r[0] if r else '')
except Exception:
    print('')" 2>/dev/null
}
CUR=$(db_run)
FLEET_UP=0
tmux has-session -t cc 2>/dev/null && tmux has-session -t cc-client 2>/dev/null && FLEET_UP=1
if [ "$CUR" != "$NAME" ]; then
  log "DB run is '${CUR:-none}' != $NAME — reset + fresh launch (seed $DONE)"
  tmux kill-session -t cc 2>/dev/null || true
  tmux kill-session -t cc-client 2>/dev/null || true
  sleep 2
  bash "$SRV/scripts/reset_fleet.sh" 3>&- || { log "reset_fleet FAILED"; exit 1; }
  env $ENV_LINE bash "$SRV/scripts/restart_fleet.sh" 3>&- || { log "restart_fleet FAILED"; exit 1; }
elif [ "$FLEET_UP" = 0 ]; then
  log "fleet down mid-trial — warm resume (seed $DONE)"
  env $ENV_LINE bash "$SRV/scripts/restart_fleet.sh" 3>&- || { log "restart_fleet FAILED"; exit 1; }
fi

# A fresh launch bootstraps the DB asynchronously; mate_bench hard-exits on a
# missing DB (the 07-20 20:25 rc=1 race). Wait for the run row before watching.
for _ in $(seq 1 24); do
  [ "$(db_run)" = "$NAME" ] && break
  sleep 5
done
if [ "$(db_run)" != "$NAME" ]; then
  log "run row for $NAME never appeared after launch — retrying next cron fire"
  exit 1
fi

# Babysitter, two failure modes the watcher can't fix itself:
#  - trainer-only SIGKILL under a live client (runs 23/24): the trial free-runs
#    into the 10h DNF bound;
#  - full fleet death mid-trial while THIS driver (not under tmux) survives.
# Normal between-trial resets keep downtime well under the strike windows.
babysit(){
  t_strikes=0; f_strikes=0
  while true; do
    sleep 120
    if pgrep -f 'lc0-clien[t]' >/dev/null && ! pgrep -f 'train_continuou[s]' >/dev/null; then
      t_strikes=$((t_strikes+1))
    else
      t_strikes=0
    fi
    if ! pgrep -f 'cc-serve[r]' >/dev/null && ! pgrep -f 'lc0-clien[t]' >/dev/null; then
      f_strikes=$((f_strikes+1))
    else
      f_strikes=0
    fi
    if [ "$t_strikes" -ge 2 ] || [ "$f_strikes" -ge 3 ]; then
      ts=$(date -u +%m%d-%H%M)
      log "FLEET DEGRADED (trainer_dead=$t_strikes fleet_dead=$f_strikes) — forensics + warm restart ($ts)"
      { tmux capture-pane -p -S -2000 -t cc:trainer 2>&1; free -m; nvidia-smi; } > "$WS/trainer-death-$ts.txt" 2>&1 || true
      tmux kill-session -t cc 2>/dev/null || true
      tmux kill-session -t cc-client 2>/dev/null || true
      sleep 2
      env $ENV_LINE bash "$SRV/scripts/restart_fleet.sh" 3>&- >> "$WS/trainer-restarts.log" 2>&1 || true
      echo "$(date -u) degraded -> warm restart ($ts)" >> "$WS/trainer-restarts.log"
      t_strikes=0; f_strikes=0
    fi
  done
}
babysit & BSPID=$!
trap 'pkill -TERM -P $BSPID 2>/dev/null; kill $BSPID 2>/dev/null || true' EXIT

cd "$ENG"
.venv/bin/python scripts/mate_bench.py --trials "$REMAINING" --max-hours 10 3>&-
rc=$?
log "mate_bench exited rc=$rc (cron resumes if trials remain)"
exit $rc
