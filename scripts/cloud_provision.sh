#!/usr/bin/env bash
# Provision a vast.ai bid instance using the canonical PyTorch (Vast)
# template (hash b84ca276fa572e949cd7ff43ae5fe855).
#
# Template bakes in:
#   - vastai/pytorch image (auto-picks the right CUDA tag for the host GPU)
#   - vast's standard entrypoint.sh — sets up the Open / Logs / Jupyter
#     buttons in the web console (the whole reason to use this template
#     over a bare pytorch/pytorch image)
#   - Jupyter on :1111, TensorBoard on :6006, SSH direct
#
# Usage:
#   scripts/cloud_provision.sh <offer_id> [bid_price]
#
# Pass the offer's own min_bid as bid_price; see feedback memory
# "vast.ai — bid exactly min_bid, never outbid".
#
# Discovery (run first to find good offers):
#   vastai search offers \
#     "cpu_cores_effective>=16 cpu_ram>=8 reliability>0.95 inet_down>=200 verified=true rentable=true gpu_ram>=4 compute_cap<1200 dph<0.20" \
#     --order min_bid --raw | python3 -c "import json, sys; ..."
set -euo pipefail

OFFER_ID="${1:?Usage: $0 <offer_id> <bid_price>}"
BID="${2:?Usage: $0 <offer_id> <bid_price>}"
TEMPLATE_HASH="${TEMPLATE_HASH:-b84ca276fa572e949cd7ff43ae5fe855}"
LABEL="${LABEL:-chessckers-cloud-workers}"

echo "Provisioning offer=$OFFER_ID at bid=\$$BID using template hash $TEMPLATE_HASH"
vastai create instance "$OFFER_ID" \
  --template_hash "$TEMPLATE_HASH" \
  --bid_price "$BID" \
  --label "$LABEL" \
  --raw
