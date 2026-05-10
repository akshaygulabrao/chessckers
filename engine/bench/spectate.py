"""All-in-one spectator: one command, one terminal, one URL.

What it does:
- Serves the static `chessground/` directory + `chessground/watch/games.jsonl`
- Implements POST /api/game/new (the only API endpoint spectate.html needs).
  Reuses the Scala server's response shape (fen/turn/check/status/stacks),
  computed in Python via PyVariantClient.
- Forks a background thread that plays NN-vs-random games (alternating
  colors) and appends each finished game to chessground/watch/games.jsonl.
  Spectate.html auto-polls that file → new games appear live as you watch.

Usage:
    uv run python bench/spectate.py
    # then open http://localhost:8080/spectate.html

Defaults: weights = runs/local-001/weights.pt, sims=200, temp=0.5, plays
forever (one game every ~30-60s on M1 Pro CPU). Hit ctrl-C to stop.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import chess
import torch

from chessckers_engine.checkpoints import load_checkpoint
from chessckers_engine.mcts_puct import run_mcts
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.random_player import pick_random
from chessckers_engine.variant_py import PyVariantClient

log = logging.getLogger("chessckers_engine.spectate")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHESSGROUND_DIR = REPO_ROOT / "chessground"
GAMES_PATH = CHESSGROUND_DIR / "watch" / "games.jsonl"


# ---- API endpoint handler ----

# Translates State.stacks (dict[int, str of pieces]) into the JSON shape
# spectate.html expects: { "a6": [{"type": "pawn", "color": "black"}, ...], ... }
# Bottom-to-top piece order matches scalachess: pieces[0] is bottom.
def _stacks_to_json(state) -> dict:
    out: dict[str, list[dict]] = {}
    for sq, pieces in state.stacks.items():
        sq_name = chess.square_name(sq)
        out[sq_name] = [
            {"type": "king" if p == "k" else "pawn", "color": "black"}
            for p in pieces
        ]
    return out


def _state_to_view_dict(client: PyVariantClient, fen: str) -> dict:
    state = client.parse(fen)
    try:
        check = bool(state.board.is_check())
    except Exception:  # noqa: BLE001
        check = False
    status, winner, _ = client.status_and_legal(state)
    return {
        "fen": fen,
        "turn": "white" if state.board.turn == chess.WHITE else "black",
        "check": check,
        "status": status,
        "winner": winner,
        "stacks": _stacks_to_json(state),
    }


class _Handler(SimpleHTTPRequestHandler):
    """Static file server for chessground/ + a single API endpoint.

    Set as a class attribute (not __init__) because BaseHTTPRequestHandler
    instances are created per-request."""
    api_client: PyVariantClient = None  # type: ignore[assignment]

    # Silence the per-request access log (still go to stderr if anything fails).
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        pass

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/game/new":
            self.send_response(404)
            self.end_headers()
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            payload = json.loads(body.decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001
            self.send_response(400)
            self.end_headers()
            return
        fen = payload.get("fen")
        try:
            view = _state_to_view_dict(self.api_client, fen) if fen else \
                   _state_to_view_dict(self.api_client,
                                       self.api_client.new_game()["fen"])
        except Exception as e:  # noqa: BLE001
            log.warning("api/game/new failed for fen=%r: %s", fen, e)
            self.send_response(500)
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return
        body = json.dumps(view).encode("utf-8")
        self.send_response(200)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        # Add CORS headers to every response so the page can fetch from elsewhere
        # if the user opened spectate.html via file:// or a different host.
        self._cors_headers()
        super().end_headers()


# ---- Game generator ----

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
    return picker, client


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


def _generator_loop(args, stop_evt: threading.Event) -> None:
    """Worker thread: play games forever, append each to GAMES_PATH."""
    GAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    arch = dict(d_hidden=args.d_hidden, c_filters=args.c_filters, n_blocks=args.n_blocks)
    last_weights_mtime = 0.0
    nn_picker = None
    nn_seed_base = int(time.time()) % 100000
    play_client = PyVariantClient()

    def random_picker(state):
        return pick_random(state.get("legalMoves") or [])

    game_idx = 0
    while not stop_evt.is_set():
        # Hot-reload weights if the file has changed (training is writing them
        # every weight_save_every steps).
        try:
            cur_mtime = Path(args.weights).stat().st_mtime
        except FileNotFoundError:
            log.info("weights file missing yet (%s) — sleeping 5s", args.weights)
            time.sleep(5)
            continue
        if cur_mtime != last_weights_mtime:
            try:
                nn_picker, _ = _build_nn_picker(
                    Path(args.weights), arch, args.device, args.sims,
                    args.temperature, nn_seed_base + game_idx,
                )
                last_weights_mtime = cur_mtime
                log.info("loaded weights @ mtime=%s", int(cur_mtime))
            except Exception as e:  # noqa: BLE001
                log.warning("weights load failed: %s — retry in 5s", e)
                time.sleep(5)
                continue

        # Alternate colors per game so we see both directions.
        if game_idx % 2 == 0:
            white, black, label = nn_picker, random_picker, "NN(W) vs random(B)"
        else:
            white, black, label = random_picker, nn_picker, "random(W) vs NN(B)"
        t0 = time.perf_counter()
        try:
            game = _play_one(white, black, play_client)
        except Exception as e:  # noqa: BLE001
            log.warning("game failed: %s — retrying in 2s", e)
            time.sleep(2)
            continue
        elapsed = time.perf_counter() - t0
        game_idx += 1
        game["iter"] = 0
        game["game_idx"] = game_idx
        game["total_games"] = 0  # unbounded
        with GAMES_PATH.open("a") as f:
            f.write(json.dumps(game) + "\n")
        log.info("game %d (%s): %s in %d plies, %.1fs",
                 game_idx, label, game["outcome"], len(game["history"]), elapsed)


# ---- Main ----

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/local-001/weights.pt")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--d-hidden", type=int, default=256)
    p.add_argument("--c-filters", type=int, default=128)
    p.add_argument("--n-blocks", type=int, default=6)
    p.add_argument("--reset-games", action="store_true",
                   help="truncate the games file at start (default: append)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")

    if args.reset_games and GAMES_PATH.exists():
        GAMES_PATH.unlink()
        log.info("cleared %s", GAMES_PATH)

    # Resolve weights to absolute BEFORE we chdir — otherwise the relative
    # path becomes wrong once we move into chessground/.
    args.weights = str(Path(args.weights).resolve())

    # Wire the per-request handler to a shared API client.
    _Handler.api_client = PyVariantClient()
    # Serve from chessground/ so spectate.html and ./watch/games.jsonl
    # resolve relative to the same root.
    import os as _os
    _os.chdir(CHESSGROUND_DIR)

    stop = threading.Event()
    gen_thread = threading.Thread(
        target=_generator_loop, args=(args, stop), daemon=True, name="game-gen",
    )
    gen_thread.start()

    addr = ("127.0.0.1", args.port)
    httpd = ThreadingHTTPServer(addr, _Handler)
    log.info("spectator ready: http://localhost:%d/spectate.html", args.port)
    log.info("(serving %s)", CHESSGROUND_DIR)
    log.info("game generator → %s", GAMES_PATH)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutdown requested")
    finally:
        stop.set()
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
