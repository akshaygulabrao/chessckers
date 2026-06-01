#!/usr/bin/env bash
# Template: launch leena's self-play workers that feed the local trainer's
# --ingest-dir via scripts/leena_sync.sh. The deployer seds the current
# semicolon-joined CHESSCKERS_START_FEN seed mix into __MIX__ before scp+run.
# Arch 256/96/4 MUST match the trainer; worker-id-base 300 -> games attribute
# to "leena"; per-worker CPU mode hot-reloads weights pushed by the sync.
cd ~/chessckers/engine
export MACHINE=leena CHESSCKERS_MAX_PLIES=80 CHESSCKERS_VALUE_DISCOUNT=0.98 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CHESSCKERS_START_FEN='__MIX__'
mkdir -p "$HOME/chessckers/run"
nohup .venv/bin/python -m chessckers_engine.selfplay_workers_only \
  --run-dir "$HOME/chessckers/run" --weights "$HOME/chessckers/run/weights.pt" \
  --workers 6 --worker-id-base 300 --device cpu --sims 400 \
  --d-hidden 256 --c-filters 96 --n-blocks 4 \
  --temperature 1.0 --dirichlet-alpha 0.5 --dirichlet-eps 0.40 \
  --max-plies 80 --weights-poll-seconds 20 --seed 4000 \
  > "$HOME/chessckers/run/workers.log" 2>&1 &
echo $! > "$HOME/chessckers/run/pid"
disown
echo "leena workers launched (pid $(cat "$HOME/chessckers/run/pid"))"
