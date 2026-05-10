#!/usr/bin/env bash
# Provision a vast.ai bid instance using our standard chessckers-cpu-workers
# template (id 411299, hash 4d455cf2147ed58a54a4f8f66230ffc8).
#
# Template bakes in:
#   - pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime image
#   - onstart that pre-installs chess==1.11.2 numpy>=1.26 httpx
#     (runs during loading phase, BEFORE the billing meter starts —
#     vast.ai only charges from actual_status=running, so any setup
#     in onstart is on vast.ai's dime)
#   - 20GB disk
#   - search filter: cpu_cores_effective>=16, cpu_ram>=8GB, net>=200Mbps,
#     reliability>0.95, gpu_ram>=4GB, compute_cap<1200 (excludes Blackwell)
#
# Usage:
#   scripts/cloud_provision.sh <offer_id> [bid_price]
#
# Discovery (run first to find good offers):
#   VAST_API_KEY="$VAST_AI_API_KEY" vastai search offers \
#     "cpu_cores_effective>=16 cpu_ram>=8 reliability>0.95 inet_down>=200 verified=true rentable=true gpu_ram>=4 compute_cap<1200 dph<0.20" \
#     --order min_bid --raw | python3 -c "import json, sys; ..."
#
# After provision: poll for ready, then SSH in to rsync engine source +
# weights and launch selfplay_workers_only.
set -euo pipefail

OFFER_ID="${1:?Usage: $0 <offer_id> [bid_price]}"
BID="${2:-0.07}"
TEMPLATE_HASH="${TEMPLATE_HASH:-4d455cf2147ed58a54a4f8f66230ffc8}"
LABEL="${LABEL:-chessckers-cloud-workers}"

KEY="$(grep '^export VAST_AI_API_KEY=' ~/.zshrc | sed 's/^export VAST_AI_API_KEY=//' | tr -d '"')"

echo "Provisioning offer=$OFFER_ID at bid=\$$BID using template hash $TEMPLATE_HASH"
VAST_API_KEY="$KEY" vastai create instance "$OFFER_ID" \
  --template_hash "$TEMPLATE_HASH" \
  --bid_price "$BID" \
  --label "$LABEL" \
  --raw
