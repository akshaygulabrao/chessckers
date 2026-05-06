#!/usr/bin/env bash
# scripts/run_4070ti.sh
#
# One-shot wrapper for the 4070 Ti exploration-heavy training run.
# Paste your vast.ai SSH string and walk away — sanity checks, code sync,
# launch, monitors, rsyncs checkpoints back, destroys the box.
#
# Usage:
#   ./scripts/run_4070ti.sh "ssh -p 12345 root@1.2.3.4"
#
# Run config (matches train_cloud.sh defaults):
#   - 100 iters × 80 games × 400 sims (4× search depth vs prior runs)
#   - 30M param network (20 blocks × 256 filters × 384 hidden)
#   - 16 worker processes (multiprocess, GIL-free)
#   - Bumped Dirichlet (α=0.5 ε=0.40) + higher final temp (0.5)
#     for Black-strategy exploration
#   - keep-best gating preserves the strongest checkpoint
#
# Estimated cost: ~$7 over ~90 hours on 4070 Ti spot ($0.075/hr)
# Estimated wall: ~3-4 days
#
# If you want to interrupt:
#   ssh -p PORT root@HOST "pkill -9 -f selfplay_az_loop"
# Then re-run this script — it'll resume via state.json.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 'ssh -p PORT root@HOST'" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SSH_STRING="$1"
export RUN_NAME="${RUN_NAME:-4070ti-explore-001}"

echo "=== 4070 Ti exploration run: $RUN_NAME ==="
echo "SSH: $SSH_STRING"
echo

# All steps share the same SSH_STRING. STEP=all chains:
# 1 sanity → 2 install+sync → 4 launch → 4-wait → 5 rsync back → 6 destroy.
# Destroy is unconditional even on failure so you don't pay for a broken run.
STEP=all "$SCRIPT_DIR/train_cloud.sh"
