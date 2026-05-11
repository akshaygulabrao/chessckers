#!/usr/bin/env bash
# Stand up the human-vs-engine play stack:
#   - Scala API on :8080 (game state authority — legal moves, FEN, terminal detection)
#   - Engine HTTP on :8082 (NN + MCTS picker, talks back to Scala for legal moves)
#   - chessground/chessckers.html (browser UI; dropdowns let you pick which
#     side a Player/Random/Material/MCTS/Engine/PUCT plays)
#
# Both servers run in tmux sessions (play_scala, play_engine). Stop with
# `scripts/stop_play.sh` or `tmux kill-session -t play_scala -t play_engine`.
#
# Engine uses CPU device so it doesn't fight the trainer's MPS.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEIGHTS="${PLAY_WEIGHTS:-$REPO_ROOT/engine/runs/local-005/weights.pt}"
PUCT_SIMS="${PLAY_PUCT_SIMS:-200}"

# Model arch must match the trained checkpoint, NOT ChesskersScorer's
# (128, 96, 4) defaults — load_checkpoint(strict=False) would silently
# leave layers at random init on shape mismatch.
D_HIDDEN="${PLAY_D_HIDDEN:-256}"
C_FILTERS="${PLAY_C_FILTERS:-128}"
N_BLOCKS="${PLAY_N_BLOCKS:-6}"

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }

# ---- preflight -----------------------------------------------------------
[ -f "$WEIGHTS" ] || { log "no weights at $WEIGHTS"; exit 1; }
command -v sbt >/dev/null || { log "sbt not found"; exit 1; }

# Kill any existing play stack before relaunching. tmux kill-session
# without a target list takes one -t at a time; loop through the names.
for sess in play_scala play_engine play_static; do
  if tmux has-session -t "$sess" 2>/dev/null; then
    log "  stopping existing $sess"
    tmux kill-session -t "$sess" 2>/dev/null || true
  fi
done

# Also kill any orphaned process still bound to the ports. This is critical:
# `tmux kill-session` only SIGHUPs the foreground tmux child; sbt/java/python
# all have a habit of orphaning to launchd. If we don't kill those here,
# the curl/nc wait-loop later races against the dying old process (it
# answers briefly, we think the new server is ready, then engine starts
# 4-5s later and finds :8080 dead because the new Scala hasn't compiled
# yet). Include 8080 — sbt's compile cache is warm, so the rebuild is fast.
for port in 8080 8081 8082; do
  pids=$(lsof -t -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
  if [ -n "$pids" ]; then
    log "  killing stale process(es) on :$port (pid: $pids)"
    kill $pids 2>/dev/null || true
    # Wait for the port to actually free up.
    for w in $(seq 1 10); do
      lsof -t -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1 || break
      sleep 1
    done
  fi
done

# ---- 1. Scala API on :8080 ----------------------------------------------
log "[1/4] starting Scala API in tmux 'play_scala' (first compile takes ~30s)"
tmux new-session -d -s play_scala "cd '$REPO_ROOT/server' && sbt run"

# Wait for :8080. First-time compile + sbt boot can be 30-90s.
for i in $(seq 1 120); do
  if curl -sf -X POST http://localhost:8080/api/game/new -H 'Content-Type: application/json' -d '{}' >/dev/null 2>&1; then
    log "  Scala API ready after ${i}s"
    break
  fi
  sleep 1
  [ "$i" -eq 120 ] && { log "Scala API didn't come up in 120s — check 'tmux a -t play_scala'"; exit 1; }
done

# ---- 2. Engine HTTP on :8082 --------------------------------------------
log "[2/4] starting engine HTTP in tmux 'play_engine'"
log "       weights=$WEIGHTS  arch=($D_HIDDEN, $C_FILTERS, $N_BLOCKS)  puct_sims=$PUCT_SIMS"
# Tee output to /tmp/play_engine.log so if the process exits and tmux
# kills the session, we can still read the error post-mortem.
ENGINE_LOG=/tmp/play_engine.log
: > "$ENGINE_LOG"
tmux new-session -d -s play_engine \
  "cd '$REPO_ROOT/engine' && \
   ENGINE_MODEL='$WEIGHTS' \
   ENGINE_DEFAULT_PICKER=puct \
   ENGINE_PUCT_SIMS=$PUCT_SIMS \
   ENGINE_D_HIDDEN=$D_HIDDEN \
   ENGINE_C_FILTERS=$C_FILTERS \
   ENGINE_N_BLOCKS=$N_BLOCKS \
   .venv/bin/python -m chessckers_engine 2>&1 | tee $ENGINE_LOG"

# Wait for :8082. Use a TCP-level probe (nc -z) — curl -sf would reject
# the 4xx that the server returns for our intentionally-malformed test
# body, even though the server is fully up and CORS-ready.
for i in $(seq 1 30); do
  if nc -z 127.0.0.1 8082 2>/dev/null; then
    log "  Engine HTTP ready after ${i}s"
    break
  fi
  sleep 1
  [ "$i" -eq 30 ] && {
    log "Engine HTTP didn't come up — last 20 lines of /tmp/play_engine.log:"
    tail -20 "$ENGINE_LOG" | sed 's/^/    /' >&2
    exit 1
  }
done

# ---- 3. Static-file server on :8081 -------------------------------------
# Chrome blocks <script type="module"> + relative imports under file://
# (CORS-style policy). Serve chessground/ via http.server so the page can
# import ./dist/chessground.js cleanly.
log "[3/4] starting static-file server in tmux 'play_static' on :8081"
tmux new-session -d -s play_static \
  "cd '$REPO_ROOT/chessground' && python3 -m http.server 8081"

for i in $(seq 1 10); do
  if curl -sf -o /dev/null http://localhost:8081/chessckers.html 2>/dev/null; then
    log "  static server ready after ${i}s"
    break
  fi
  sleep 1
  [ "$i" -eq 10 ] && { log "static server didn't come up — check 'tmux a -t play_static'"; exit 1; }
done

# ---- 4. Print URL ------------------------------------------------------
URL="http://localhost:8081/chessckers.html"
log "[4/4] DONE"
log ""
log "  ┌──────────────────────────────────────────────────────────────┐"
log "  │  $URL              │"
log "  └──────────────────────────────────────────────────────────────┘"

# Clipboard the URL too — pbcopy if available (macOS), no-op otherwise.
if command -v pbcopy >/dev/null; then
  printf '%s' "$URL" | pbcopy
  log "  (URL copied to clipboard)"
fi

log ""
log "  In the UI, choose 'PUCT' (strongest, MCTS + NN) or 'Engine' (raw NN,"
log "  faster but weaker) for whichever side you want the network to play."
log ""
log "  Stop the play stack:"
log "    tmux kill-session -t play_scala -t play_engine -t play_static 2>/dev/null"
log "  Live logs:"
log "    tmux a -t play_scala       # Scala API"
log "    tmux a -t play_engine      # Engine HTTP + MCTS"
log "    tmux a -t play_static      # browser asset server"
