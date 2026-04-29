"""Smoke test: play one self-play game with a JsonlWatchSink pointed at
/tmp/spectate-smoke. Verify current.json updates per move and games.jsonl
gets one finished game appended.

Requires the scalachess server running on localhost:8080.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch

from chessckers_engine.checkpoints import load_checkpoint
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.selfplay_az import JsonlWatchSink, play_az_game
from chessckers_engine.server_client import ServerClient

WATCH_DIR = Path("/tmp/spectate-smoke")
if WATCH_DIR.exists():
    shutil.rmtree(WATCH_DIR)


def main() -> int:
    client = ServerClient()
    try:
        client.new_game()
    except Exception as e:  # noqa: BLE001
        print(f"server not reachable: {e}")
        return 1

    model = ChesskersScorer()
    ckpt = Path(__file__).resolve().parent / "weights/ln/iter-az-005.pt"
    if ckpt.exists():
        load_checkpoint(model, ckpt)
    model.eval()

    sink = JsonlWatchSink(WATCH_DIR)
    g = torch.Generator().manual_seed(0)
    game = play_az_game(
        model, client,
        n_sims=10, c_puct=1.5, temperature=0.5, rng=g,
        dirichlet_alpha=0.3, dirichlet_eps=0.25,
        sink=sink, sink_context={"iter": 1, "game_idx": 1, "total_iters": 1, "total_games": 1},
    )
    client.close()

    print(f"game outcome: {game.outcome}, plies: {len(game.records)}")
    print(f"watch dir contents: {sorted(p.name for p in WATCH_DIR.iterdir())}")

    cur = json.loads((WATCH_DIR / "current.json").read_text())
    print(f"current.json keys: {sorted(cur.keys())}")
    print(f"current.json ply={cur['ply']} last_uci={cur['last_uci']}")

    games_lines = (WATCH_DIR / "games.jsonl").read_text().splitlines()
    print(f"games.jsonl line count: {len(games_lines)}")
    rec = json.loads(games_lines[0])
    print(f"games.jsonl[0] keys: {sorted(rec.keys())}")
    print(f"games.jsonl[0] outcome={rec['outcome']} controllers={rec['controllers']} "
          f"history_len={len(rec['history'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
