#!/usr/bin/env bash
# Create ONE cheap GCP Spot self-play node and point it at THIS Mac's Tailscale trainer. Grants
# the default Compute SA read access to the ts-selfplay-authkey secret, then boots a Debian box
# that self-provisions via scripts/gcp/startup.sh (Tailscale join + deps + public clone + client).
# Re-runnable for more nodes: NAME=chessckers-sp-2 scripts/gcp/create-node.sh
#
# Env knobs: PROJECT ZONE NAME MACHINE TRAINER_IP  (sensible defaults below).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${ZONE:-us-central1-a}"
NAME="${NAME:-chessckers-sp-1}"
MACHINE="${MACHINE:-e2-medium}"        # 2 vCPU / 4GB — torch-CPU self-play floor without OOM
TRAINER_IP="${TRAINER_IP:-$(tailscale ip -4 2>/dev/null | head -1)}"
[ -n "$TRAINER_IP" ] || { echo "no Tailscale IP on this Mac — run 'tailscale up' first"; exit 1; }

PNUM="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
SA="${PNUM}-compute@developer.gserviceaccount.com"
echo "[gcp] grant $SA read on secret ts-selfplay-authkey…"
gcloud secrets add-iam-policy-binding ts-selfplay-authkey \
  --project="$PROJECT" --member="serviceAccount:$SA" \
  --role="roles/secretmanager.secretAccessor" >/dev/null

echo "[gcp] create $NAME ($MACHINE Spot, $ZONE) -> trainer ${TRAINER_IP}:8000…"
gcloud compute instances create "$NAME" \
  --project="$PROJECT" --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=20GB --boot-disk-type=pd-standard \
  --scopes=cloud-platform \
  --tags=chessckers-sp \
  --metadata=trainer-ip="$TRAINER_IP" \
  --metadata-from-file=startup-script="$REPO_ROOT/scripts/gcp/startup.sh"

cat <<EOF

[gcp] $NAME launching. Tailscale join + deps + build take a few minutes. Watch it:
  gcloud compute instances get-serial-port-output $NAME --zone $ZONE | tail -50
Expect it in:   tailscale status        (as $NAME)
Then the server tab logs /next_game + /upload_game from client gcp-$NAME.
Tear down:      gcloud compute instances delete $NAME --zone $ZONE
EOF
