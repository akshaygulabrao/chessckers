#!/usr/bin/env bash
# install_monitor_crons.sh — add monitoring cron jobs to root's crontab on the box.
# Idempotent: each line is guarded by a grep pattern; already-present lines are skipped.
# Run this ON THE BOX (deploy via `cc ssh bash /path/to/install_monitor_crons.sh`).
set -e

# Games/pair comes from champ_ladder.py's --games default (40). If late-run game
# lengths blow the nightly window (04:45 + ~2-4h), pin a smaller --games here.
CHAMPS_LINE='45 4 * * * cd /workspace/chessckers/engine && flock -n /tmp/champs_audit.lock nice -n 10 .venv/bin/python scripts/champ_ladder.py --jsonl /workspace/chessckers/lczero-server/trainer/run1/champs_audit.jsonl >> /workspace/champs_cron.log 2>&1'
CHAMPS_GUARD='champ_ladder.py'

install_line() {
    local guard="$1"
    local line="$2"
    local desc="$3"
    if crontab -l 2>/dev/null | grep -qF "$guard"; then
        echo "[cron] already present: $desc"
    else
        ( crontab -l 2>/dev/null; echo "$line" ) | crontab -
        echo "[cron] added: $desc"
    fi
}

install_line "$CHAMPS_GUARD" "$CHAMPS_LINE" "daily champs audit at 04:45"

echo "[cron] done. Current crontab:"
crontab -l 2>/dev/null
