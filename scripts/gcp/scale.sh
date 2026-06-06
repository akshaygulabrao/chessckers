#!/usr/bin/env bash
# Scale the chessckers self-play MIG to N instances. N=0 -> zero cost (no instances/disks/IPs).
# Each e2-standard-8 = 8 self-play vCPU feeding the trainer over Tailscale. Manual resize is the
# right knob here: self-play boxes are always 100% CPU by design, so there's no autoscaling
# metric to chase — you set the spend, GCP fills it from Spot capacity across the region.
set -euo pipefail
N="${1:?usage: scale.sh <N>   (number of self-play instances; 0 = off)}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
MIG="${MIG:-chessckers-sp-mig}"
gcloud compute instance-groups managed resize "$MIG" \
  --project="$PROJECT" --region="$REGION" --size="$N"
echo "[mig] $MIG -> size $N.  watch:  gcloud compute instance-groups managed list-instances $MIG --region $REGION"
