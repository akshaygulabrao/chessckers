"""Generate a single self-contained HTML viewer with games embedded.

Output: one .html file you double-click. No server, no internet, no
external deps. Plays games against random with the current weights, then
serializes them + a minimal SVG board renderer + ply controls into the
output HTML.

Usage:
    uv run python bench/make_spectate_html.py
    # → writes spectate.html, then `open spectate.html`
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

import torch

from chessckers_engine.checkpoints import load_checkpoint
from chessckers_engine.mcts_puct import run_mcts
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.random_player import pick_random
from chessckers_engine.variant_py import PyVariantClient

log = logging.getLogger("chessckers_engine.spectate_html")


# ---- Game generator (same shape as spectate.py / watch_game.py) ----

def _build_nn_picker(weights: Path, arch: dict, device: str, sims: int,
                     temp: float, seed: int):
    model = ChesskersScorer(**arch).to(device)
    load_checkpoint(model, weights)
    model.eval()
    client = PyVariantClient()
    rng = random.Random(seed)

    def picker(state):
        result = run_mcts(state, client, model, n_sims=sims)
        if not result.visit_distribution or result.chosen is None:
            return result.chosen
        if temp <= 0:
            return result.chosen
        ucis = list(result.visit_distribution.keys())
        visits = [result.visit_distribution[u] for u in ucis]
        invT = 1.0 / temp
        weights_ = [v ** invT for v in visits]
        s = sum(weights_)
        if s <= 0:
            return result.chosen
        probs = [w / s for w in weights_]
        chosen_uci = rng.choices(ucis, weights=probs, k=1)[0]
        for uci, child in result.root.children.items():
            if uci == chosen_uci:
                return child.move_to_here
        return result.chosen
    return picker


def _play_one(white_picker, black_picker, client: PyVariantClient,
              max_plies: int = 400) -> dict:
    state = client.new_game()
    history = []
    ply = 0
    while not state.get("status") and ply < max_plies:
        cur_fen = state["fen"]
        picker = white_picker if state["turn"] == "white" else black_picker
        move = picker(state)
        if move is None:
            break
        state = client.make_move(cur_fen, move["uci"])
        history.append({"fen": cur_fen, "uci": move["uci"]})
        ply += 1
    final_fen = state["fen"]
    if state.get("status"):
        outcome = state.get("winner") or state["status"]
    elif ply >= max_plies:
        outcome = "draw-max-plies"
    else:
        outcome = "incomplete"
    return {"history": history, "final_fen": final_fen, "outcome": outcome}


# ---- HTML template ----
#
# All-in-one viewer: parses Chessckers FEN client-side, renders an SVG
# board, steps through plies. No external assets.

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Chessckers — game viewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #1a1a2e; color: #eee; min-height: 100vh; padding: 20px;
    display: grid; grid-template-columns: minmax(0, 1fr) 280px; gap: 20px;
    max-width: 1100px; margin: 0 auto;
  }
  h1 { font-size: 1.4em; margin-bottom: 4px; }
  .subtitle { color: #888; font-size: 0.85em; margin-bottom: 14px; }
  #board { width: min(70vw, 640px); aspect-ratio: 1; }
  rect.light { fill: #efdfb6; }
  rect.dark { fill: #b1855e; }
  rect.last-move { stroke: #ffeb3b; stroke-width: 2.5; }
  .file-label, .rank-label {
    font-size: 9px; fill: #5a4a3a; pointer-events: none;
    font-family: -apple-system, sans-serif;
  }
  .stack-badge {
    font-family: -apple-system, sans-serif; font-weight: bold;
    fill: #e63946; stroke: #000; stroke-width: 0.6;
  }
  #controls { margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  button { padding: 7px 14px; border: 1px solid #2a3358; border-radius: 5px;
           background: #16213e; color: #eee; cursor: pointer; font-size: 0.9em; }
  button:hover { background: #1f2d54; }
  button.primary { background: #2a6f4a; border-color: #2a6f4a; }
  button.primary:hover { background: #348a5a; }
  input[type="range"] { flex: 1; min-width: 200px; }
  .panel { background: #16213e; border: 1px solid #2a3358; border-radius: 6px;
           padding: 12px; margin-bottom: 12px; }
  .panel h2 { font-size: 0.95em; margin-bottom: 8px; color: #aaa; }
  .row { display: flex; justify-content: space-between; padding: 3px 0;
         font-size: 0.85em; border-bottom: 1px solid #1f2d54; }
  .row:last-child { border: none; }
  .k { color: #888; } .v { color: #eee; font-weight: 500; }
  .v.white { color: #fce4a8; } .v.black { color: #ff6b6b; } .v.draw { color: #aaa; }
  select { width: 100%; padding: 6px; background: #0f1626; color: #eee;
           border: 1px solid #2a3358; border-radius: 4px; font-size: 0.85em; }
  code { background: #0f1626; padding: 1px 6px; border-radius: 3px; font-size: 0.85em; }
</style>
</head>
<body>
  <div>
    <h1>Chessckers — game viewer</h1>
    <div class="subtitle">__SUBTITLE__</div>
    <svg id="board" viewBox="0 0 800 800"></svg>
    <div id="controls">
      <button id="prev-game">← prev game</button>
      <button id="next-game">next game →</button>
      <button id="restart">⟲</button>
      <button id="prev-ply">◀</button>
      <button id="play-pause" class="primary">▶ play</button>
      <button id="next-ply">▶</button>
      <input id="ply-slider" type="range" min="0" max="0" value="0" />
    </div>
  </div>
  <aside>
    <div class="panel">
      <h2>game</h2>
      <select id="game-select"></select>
      <div class="row" style="margin-top:8px"><span class="k">outcome</span><span class="v" id="outcome">—</span></div>
      <div class="row"><span class="k">ply</span><span class="v" id="ply">0 / 0</span></div>
      <div class="row"><span class="k">turn</span><span class="v" id="turn">—</span></div>
      <div class="row"><span class="k">last move</span><span class="v"><code id="last-uci">—</code></span></div>
    </div>
    <div class="panel">
      <h2>about</h2>
      <div class="row"><span class="k">white pieces</span><span class="v">PNBRQK glyphs</span></div>
      <div class="row"><span class="k">black stones</span><span class="v" style="color:#e63946">⬤ disk</span></div>
      <div class="row"><span class="k">black kings</span><span class="v" style="color:#e63946">♛</span></div>
      <div class="row"><span class="k">stack height</span><span class="v">red badge bottom-right</span></div>
    </div>
  </aside>
<script>
  const GAMES = __GAMES_JSON__;
  const TEMPO_MS = 700;

  const SQ = 80;
  const PIECE_COLOR_WHITE = '#f8f4eb';
  const PIECE_COLOR_BLACK_STROKE = '#1a1a1a';

  // Unicode glyphs for white pieces. We render them in an SVG <text>.
  const WHITE_GLYPH = { P: '♙', N: '♘', B: '♗', R: '♖', Q: '♕', K: '♔' };

  function parseFEN(fen) {
    // Returns {board: 8x8 of {color, type} | null, turn, stacks: {sq: pieces}}
    // FEN forms used by the engine:
    //   "<rows> [<overlay>] <turn> <castle> <ep> <half> <full>"
    //   "<rows> <turn> ..."
    let rows, overlay = '', rest;
    const m = fen.match(/^([^\s\[]+)(?:\[([^\]]*)\])?\s+(.*)$/);
    if (!m) return null;
    rows = m[1]; overlay = m[2] || ''; rest = m[3].trim();
    const turn = rest.split(/\s+/)[0] === 'b' ? 'black' : 'white';

    const board = Array.from({length: 8}, () => Array(8).fill(null));
    const ranks = rows.split('/');
    for (let i = 0; i < 8; i++) {
      const y = 7 - i;
      let x = 0;
      for (const ch of ranks[i]) {
        if (/\d/.test(ch)) { x += +ch; continue; }
        if (ch >= 'A' && ch <= 'Z') {
          board[y][x] = { color: 'white', type: ch };
        } else if (ch === 'p') {
          board[y][x] = { color: 'black', type: 'stone' };
        } else if (ch === 'k') {
          board[y][x] = { color: 'black', type: 'king' };
        }
        x += 1;
      }
    }
    const stacks = {};
    if (overlay) {
      for (const entry of overlay.split(',')) {
        const t = entry.trim();
        if (!t.includes(':')) continue;
        const [sq, pieces] = t.split(':');
        stacks[sq] = pieces;
      }
    }
    return { board, turn, stacks };
  }

  function sqName(x, y) {
    return String.fromCharCode(97 + x) + (y + 1);
  }
  function fromUCI(uci) {
    if (!uci || uci.length < 4) return null;
    const a = uci.slice(0, 2), b = uci.slice(2, 4);
    return [a, b];
  }

  function render(fen, lastUci) {
    const svg = document.getElementById('board');
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    const parsed = parseFEN(fen);
    if (!parsed) return;
    const { board, stacks } = parsed;
    const lm = fromUCI(lastUci);

    for (let y = 7; y >= 0; y--) {
      for (let x = 0; x < 8; x++) {
        const px = x * SQ;
        const py = (7 - y) * SQ;
        const isLight = (x + y) % 2 === 1;
        const sn = sqName(x, y);
        const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        r.setAttribute('x', px); r.setAttribute('y', py);
        r.setAttribute('width', SQ); r.setAttribute('height', SQ);
        r.setAttribute('class', isLight ? 'light' : 'dark');
        if (lm && (lm[0] === sn || lm[1] === sn)) {
          r.classList.add('last-move');
        }
        svg.appendChild(r);

        // file/rank labels in the corner
        if (x === 0) {
          const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
          t.setAttribute('x', px + 3); t.setAttribute('y', py + 11);
          t.setAttribute('class', 'rank-label'); t.textContent = (y + 1).toString();
          svg.appendChild(t);
        }
        if (y === 0) {
          const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
          t.setAttribute('x', px + SQ - 9); t.setAttribute('y', py + SQ - 4);
          t.setAttribute('class', 'file-label');
          t.textContent = String.fromCharCode(97 + x);
          svg.appendChild(t);
        }

        const cell = board[y][x];
        if (cell) {
          if (cell.color === 'white') {
            const g = WHITE_GLYPH[cell.type];
            const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            t.setAttribute('x', px + SQ / 2); t.setAttribute('y', py + SQ / 2 + 22);
            t.setAttribute('text-anchor', 'middle');
            t.setAttribute('font-size', '60');
            t.setAttribute('fill', PIECE_COLOR_WHITE);
            t.setAttribute('stroke', PIECE_COLOR_BLACK_STROKE);
            t.setAttribute('stroke-width', '1.2');
            t.textContent = g || '?';
            svg.appendChild(t);
          } else {
            // Black: red disk with optional crown for kings.
            const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            c.setAttribute('cx', px + SQ / 2); c.setAttribute('cy', py + SQ / 2);
            c.setAttribute('r', SQ * 0.36);
            c.setAttribute('fill', '#e63946');
            c.setAttribute('stroke', '#1a1a1a');
            c.setAttribute('stroke-width', '2');
            svg.appendChild(c);
            if (cell.type === 'king') {
              const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
              t.setAttribute('x', px + SQ / 2); t.setAttribute('y', py + SQ / 2 + 12);
              t.setAttribute('text-anchor', 'middle');
              t.setAttribute('font-size', '32');
              t.setAttribute('fill', '#fff');
              t.setAttribute('stroke', '#000');
              t.setAttribute('stroke-width', '0.5');
              t.textContent = '♛';
              svg.appendChild(t);
            }
          }
        }

        // Stack-height badge for towers > 1.
        const stack = stacks[sn];
        if (stack && stack.length > 1) {
          const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
          t.setAttribute('x', px + SQ - 4); t.setAttribute('y', py + SQ - 4);
          t.setAttribute('text-anchor', 'end');
          t.setAttribute('font-size', '20');
          t.setAttribute('class', 'stack-badge');
          t.textContent = stack.length.toString();
          svg.appendChild(t);
          // King markers: count 'k' in stack and show as e.g. "k1,2"
          const kingPositions = [];
          for (let idx = 0; idx < stack.length; idx++) {
            if (stack[idx] === 'k') kingPositions.push(stack.length - idx);
          }
          if (kingPositions.length > 0) {
            const kt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            kt.setAttribute('x', px + SQ - 4); kt.setAttribute('y', py + SQ - 22);
            kt.setAttribute('text-anchor', 'end');
            kt.setAttribute('font-size', '12');
            kt.setAttribute('class', 'stack-badge');
            kt.textContent = 'k' + kingPositions.sort((a,b)=>a-b).join(',');
            svg.appendChild(kt);
          }
        }
      }
    }
  }

  // ---- state ----
  let gameIdx = 0;
  let plyIdx = 0;
  let plies = [];
  let playing = false;
  let timer = null;
  const $ = (id) => document.getElementById(id);

  function pliesFor(g) {
    const fens = (g.history || []).map(h => h.fen);
    if (g.final_fen) fens.push(g.final_fen);
    return fens;
  }
  function uciAt(g, idx) {
    if (idx <= 0) return null;
    const h = g.history || [];
    return idx - 1 < h.length ? h[idx - 1].uci : null;
  }

  function loadGame(idx) {
    if (idx < 0 || idx >= GAMES.length) return;
    gameIdx = idx;
    const g = GAMES[idx];
    plies = pliesFor(g);
    plyIdx = 0;
    $('outcome').textContent = g.outcome || '—';
    $('outcome').className = 'v ' + (g.outcome || 'draw');
    $('ply-slider').max = String(Math.max(0, plies.length - 1));
    $('game-select').value = String(idx);
    renderPly();
  }

  function renderPly() {
    if (!plies.length) return;
    const fen = plies[plyIdx];
    const luc = uciAt(GAMES[gameIdx], plyIdx);
    render(fen, luc);
    const turn = parseFEN(fen).turn;
    $('turn').textContent = turn;
    $('turn').className = 'v ' + turn;
    $('ply').textContent = `${plyIdx} / ${plies.length - 1}`;
    $('last-uci').textContent = luc || '(start)';
    $('ply-slider').value = String(plyIdx);
  }

  function setPlaying(p) {
    playing = p;
    $('play-pause').textContent = playing ? '⏸ pause' : '▶ play';
    if (timer) { clearInterval(timer); timer = null; }
    if (playing) {
      timer = setInterval(() => {
        if (plyIdx < plies.length - 1) {
          plyIdx += 1;
          renderPly();
        } else {
          setPlaying(false);
        }
      }, TEMPO_MS);
    }
  }

  // controls
  $('prev-ply').onclick = () => { if (plyIdx > 0) { plyIdx -= 1; renderPly(); setPlaying(false); } };
  $('next-ply').onclick = () => { if (plyIdx < plies.length - 1) { plyIdx += 1; renderPly(); setPlaying(false); } };
  $('prev-game').onclick = () => loadGame(gameIdx - 1);
  $('next-game').onclick = () => loadGame(gameIdx + 1);
  $('restart').onclick = () => { plyIdx = 0; renderPly(); setPlaying(true); };
  $('play-pause').onclick = () => setPlaying(!playing);
  $('ply-slider').oninput = (e) => { plyIdx = parseInt(e.target.value, 10); renderPly(); setPlaying(false); };
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') $('prev-ply').click();
    else if (e.key === 'ArrowRight') $('next-ply').click();
    else if (e.key === ' ') { e.preventDefault(); $('play-pause').click(); }
  });

  // game selector
  const sel = $('game-select');
  GAMES.forEach((g, i) => {
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = `game ${i+1}: ${g.label || ''} → ${g.outcome || '?'}`;
    sel.appendChild(opt);
  });
  sel.onchange = () => loadGame(+sel.value);

  loadGame(0);
  setPlaying(true);
</script>
</body>
</html>
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path,
                   default=Path("runs/local-001/weights.pt"))
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--d-hidden", type=int, default=256)
    p.add_argument("--c-filters", type=int, default=128)
    p.add_argument("--n-blocks", type=int, default=6)
    p.add_argument("--out", type=Path, default=Path("spectate.html"))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")

    arch = dict(d_hidden=args.d_hidden, c_filters=args.c_filters,
                n_blocks=args.n_blocks)
    nn_picker = _build_nn_picker(args.weights, arch, args.device, args.sims,
                                 args.temperature, args.seed)

    def random_picker(state):
        return pick_random(state.get("legalMoves") or [])

    play_client = PyVariantClient()
    games = []
    for i in range(args.games):
        if i % 2 == 0:
            white, black, label = nn_picker, random_picker, "NN(W) vs random(B)"
        else:
            white, black, label = random_picker, nn_picker, "random(W) vs NN(B)"
        t0 = time.perf_counter()
        g = _play_one(white, black, play_client)
        elapsed = time.perf_counter() - t0
        g["label"] = label
        games.append(g)
        log.info("game %d/%d (%s): %s in %d plies, %.1fs",
                 i + 1, args.games, label, g["outcome"], len(g["history"]), elapsed)

    # Inline-embed the games as JSON in the HTML.
    games_json = json.dumps(games)
    subtitle = (f"{args.games} game(s) — weights={args.weights}, "
                f"sims={args.sims}, temp={args.temperature}")
    page = HTML_TEMPLATE.replace("__GAMES_JSON__", games_json) \
                        .replace("__SUBTITLE__", html.escape(subtitle))
    args.out.write_text(page)
    log.info("wrote %s (%.1f KB)", args.out, args.out.stat().st_size / 1024)
    log.info("open in browser: open %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
