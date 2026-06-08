#!/usr/bin/env bash
# GCE startup-script: provision a headless Linux self-play CLIENT and join it to the Mac trainer
# over Tailscale. Runs as root on every boot (idempotent-ish). Output streams to the serial
# console (gcloud compute instances get-serial-port-output) AND /var/log/chessckers-startup.log.
# Reusable verbatim as the MIG instance-template startup-script.
#
#   metadata: trainer-ip   = the Mac trainer's Tailscale 100.x address
#   secret:   ts-selfplay-authkey (Secret Manager) = reusable+ephemeral Tailscale auth key,
#             read via the instance service-account token (no gcloud dependency); needs
#             cloud-platform scope + secretAccessor IAM (both set by create-node.sh).
set -uxo pipefail
exec > >(tee -a /var/log/chessckers-startup.log) 2>&1
echo "=== chessckers-sp startup $(date -u) ==="

# Base tools FIRST (curl/python3 are used immediately below; not guaranteed on a bare image).
# python3-dev is REQUIRED: uv builds the venv on the system python3, and cpp/build.sh's
# `find_package(Python COMPONENTS Development.Module)` needs Python.h (absent without it) — the
# native build fails at cmake-configure otherwise (verified on a fresh debian-12 box, 2026-06-07).
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y curl ca-certificates python3 python3-dev git build-essential cmake pkg-config libopenblas-dev

md(){ curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/$1"; }
TRAINER_IP="$(md instance/attributes/trainer-ip)"
PROJECT="$(md project/project-id)"
SERVER="http://${TRAINER_IP}:8000"
echo "trainer=$SERVER project=$PROJECT"

# --- Tailscale: install + join the tailnet ---
command -v tailscale >/dev/null 2>&1 || curl -fsSL https://tailscale.com/install.sh | sh
TOKEN="$(md instance/service-accounts/default/token | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')"
AUTHKEY="$(curl -s -H "Authorization: Bearer $TOKEN" \
  "https://secretmanager.googleapis.com/v1/projects/${PROJECT}/secrets/ts-selfplay-authkey/versions/latest:access" \
  | python3 -c 'import sys,json,base64;print(base64.b64decode(json.load(sys.stdin)["payload"]["data"]).decode().strip())')"
tailscale up --authkey="$AUTHKEY" --hostname="$(hostname)" --accept-dns=true
echo "tailnet IP: $(tailscale ip -4 2>/dev/null | head -1)  ->  trainer $SERVER"

# --- app user: code + venv (CPU torch) + native engine ---
# Native C++ engine on Linux (lc0-split): the NN forward is portable cblas_sgemm, so this box
# builds the SAME cc_selfplay engine as the Apple boxes — against OpenBLAS (libopenblas-dev
# above) instead of Accelerate. No Python self-play fallback. cpp/build.sh builds chessckers_cpp
# + cc_selfplay; the systemd client (launch_gcp.sh) runs cc_selfplay --jobs-local.
id -u sp >/dev/null 2>&1 || useradd -m -s /bin/bash sp
runuser -u sp -- bash <<'USR'
set -uxo pipefail
export HOME=/home/sp
cd "$HOME"
curl -LsSf https://astral.sh/uv/install.sh | sh                       # uv (Python pkg mgr)
export PATH="$HOME/.local/bin:$PATH"
[ -d chessckers ] || git clone --depth=1 https://github.com/akshaygulabrao/chessckers.git chessckers
cd chessckers/engine
UV_TORCH_BACKEND=cpu uv sync          # force the CPU torch wheel (Linux default is the ~3GB CUDA build)
PATH="$HOME/chessckers/engine/.venv/bin:$PATH" cpp/build.sh   # native engine (chessckers_cpp + cc_selfplay, OpenBLAS)
USR

# --- run the client as a systemd service (survives startup-script exit; restarts on crash) ---
cat > /etc/systemd/system/chessckers-sp.service <<UNIT
[Unit]
Description=Chessckers self-play client
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
User=sp
WorkingDirectory=/home/sp/chessckers
Environment=SERVER=${SERVER}
Environment=PATH=/home/sp/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/sp/chessckers/scripts/launch_gcp.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now chessckers-sp.service
echo "=== startup done; chessckers-sp.service up -> $SERVER ==="
