#!/usr/bin/env bash
# Create the instance TEMPLATE + a regional MIG for elastic 0..N chessckers self-play. The
# template bakes scripts/gcp/startup.sh (Tailscale join + deps + public clone + client) so every
# instance self-provisions identically; per-node worker-id-base (launch_gcp.sh hostname hash)
# keeps their self-play streams decorrelated. Spot + STOP-on-preempt (MIG-required) => the MIG
# auto-recreates preempted boxes to hold target size; an ephemeral Tailscale key deregisters the
# dead nodes. Regional
# => Spot capacity is found across every us-central1 zone. Starts at size 0 (zero cost); scale
# with scripts/gcp/scale.sh.
#
# Templates are IMMUTABLE: the name is versioned by a hash of startup.sh + machine type, so any
# startup.sh edit makes a new template and rolls the MIG onto it; an unchanged re-run is a no-op.
# (The engine + launch_gcp.sh are cloned fresh from public origin at each boot, so ordinary code
# changes need no new template — just recycle instances or let preemption recycle them.)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
MIG="${MIG:-chessckers-sp-mig}"
MACHINE="${MACHINE:-e2-standard-8}"
TRAINER_IP="${TRAINER_IP:-$(tailscale ip -4 2>/dev/null | head -1)}"
[ -n "$TRAINER_IP" ] || { echo "no Tailscale IP on this Mac — run 'tailscale up' first"; exit 1; }

TPL_HASH="$(shasum "$REPO_ROOT/scripts/gcp/startup.sh" | cut -c1-8)"
TEMPLATE="${MIG}-${MACHINE//[^a-z0-9]/-}-${TPL_HASH}"

PNUM="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
SA="${PNUM}-compute@developer.gserviceaccount.com"
echo "[mig] grant $SA read on secret ts-selfplay-authkey…"
gcloud secrets add-iam-policy-binding ts-selfplay-authkey \
  --project="$PROJECT" --member="serviceAccount:$SA" \
  --role="roles/secretmanager.secretAccessor" >/dev/null

# 1. instance template (immutable; create only if this exact version is absent).
if ! gcloud compute instance-templates describe "$TEMPLATE" --project="$PROJECT" >/dev/null 2>&1; then
  echo "[mig] create template $TEMPLATE ($MACHINE Spot)…"
  gcloud compute instance-templates create "$TEMPLATE" \
    --project="$PROJECT" \
    --machine-type="$MACHINE" \
    --provisioning-model=SPOT --instance-termination-action=STOP \
    --image-family=debian-12 --image-project=debian-cloud \
    --boot-disk-size=20GB --boot-disk-type=pd-standard \
    --scopes=cloud-platform \
    --tags=chessckers-sp \
    --metadata=trainer-ip="$TRAINER_IP" \
    --metadata-from-file=startup-script="$REPO_ROOT/scripts/gcp/startup.sh"
else
  echo "[mig] template $TEMPLATE already exists — reusing."
fi

# 2. regional MIG: create at size 0 if absent; else roll onto the (new) template when it differs.
if ! gcloud compute instance-groups managed describe "$MIG" --project="$PROJECT" --region="$REGION" >/dev/null 2>&1; then
  echo "[mig] create regional MIG $MIG in $REGION at size 0…"
  gcloud compute instance-groups managed create "$MIG" \
    --project="$PROJECT" --region="$REGION" \
    --template="$TEMPLATE" --size=0
else
  CUR_TPL="$(gcloud compute instance-groups managed describe "$MIG" --project="$PROJECT" --region="$REGION" \
    --format='value(instanceTemplate)' | sed 's#.*/##')"
  if [ "$CUR_TPL" != "$TEMPLATE" ]; then
    echo "[mig] roll $MIG: $CUR_TPL -> $TEMPLATE…"
    gcloud compute instance-groups managed set-instance-template "$MIG" \
      --project="$PROJECT" --region="$REGION" --template="$TEMPLATE"
    gcloud compute instance-groups managed rolling-action replace "$MIG" \
      --project="$PROJECT" --region="$REGION" --max-unavailable=100%
  else
    echo "[mig] $MIG already on $TEMPLATE — nothing to roll."
  fi
fi

cat <<EOF

[mig] ready. $MIG ($REGION) at size 0 -> trainer ${TRAINER_IP}:8000.
  scale up:    scripts/gcp/scale.sh 4      # 4 x $MACHINE = 32 self-play vCPU
  scale down:  scripts/gcp/scale.sh 0      # back to zero cost
  list nodes:  gcloud compute instance-groups managed list-instances $MIG --region $REGION
EOF
