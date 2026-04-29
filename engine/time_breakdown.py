"""Profile one self-play game: how much time goes to HTTP, NN forward, and
the PUCT logic in between?

Monkey-patches `ServerClient._post` and `ChesskersScorer.{forward, value,
policy_and_value}` to accumulate wall time + call counts, plays one game,
and reports a breakdown.

Run: `uv run python time_breakdown.py [SIMS]`  (default 25 sims/move)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

from chessckers_engine.checkpoints import load_checkpoint
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.selfplay_az import play_az_game
from chessckers_engine.server_client import ServerClient


class T:
    http_t = 0.0
    http_n = 0
    nn_t = 0.0
    nn_n = 0


def _wrap(orig, accum_attr_t: str, accum_attr_n: str):
    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            setattr(T, accum_attr_t, getattr(T, accum_attr_t) + (time.perf_counter() - t0))
            setattr(T, accum_attr_n, getattr(T, accum_attr_n) + 1)
    return wrapped


# Patch HTTP
ServerClient._post = _wrap(ServerClient._post, "http_t", "http_n")

# Patch NN heads (all three entry points used during MCTS + selfplay)
ChesskersScorer.forward = _wrap(ChesskersScorer.forward, "nn_t", "nn_n")
ChesskersScorer.value = _wrap(ChesskersScorer.value, "nn_t", "nn_n")
ChesskersScorer.policy_and_value = _wrap(ChesskersScorer.policy_and_value, "nn_t", "nn_n")


def main() -> int:
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    ckpt = Path(__file__).resolve().parent / "weights/ln/iter-az-005.pt"

    client = ServerClient()
    try:
        client.new_game()
    except Exception as e:  # noqa: BLE001
        print(f"server not reachable: {e}")
        return 1

    model = ChesskersScorer()
    if ckpt.exists():
        load_checkpoint(model, ckpt)
    model.eval()

    g = torch.Generator().manual_seed(0)
    print(f"profiling 1 self-play game @ {n_sims} sims/move…")
    wall = time.perf_counter()
    game = play_az_game(
        model, client,
        n_sims=n_sims, c_puct=1.5, temperature=0.5, rng=g,
        dirichlet_alpha=0.3, dirichlet_eps=0.25,
    )
    wall = time.perf_counter() - wall
    client.close()

    plies = len(game.records)
    other_t = wall - T.http_t - T.nn_t

    def row(label: str, t: float, n: int) -> str:
        pct = 100.0 * t / wall if wall > 0 else 0.0
        avg = (t / n * 1000) if n else 0.0
        return f"  {label:<10s} {t*1000:8.0f} ms  {pct:5.1f}%  {n:6d} calls  ({avg:6.2f} ms/call avg)"

    print(f"\noutcome: {game.outcome}, plies: {plies}")
    print(f"\nTotal wall: {wall*1000:.0f} ms ({wall/plies*1000:.0f} ms/ply)")
    print(row("http", T.http_t, T.http_n))
    print(row("nn",   T.nn_t,   T.nn_n))
    print(f"  other      {other_t*1000:8.0f} ms  {100*other_t/wall:5.1f}%  (PUCT logic, encoding, Python overhead)")

    # Estimated speedups
    if T.http_t > 0:
        print("\nIf HTTP went to ~zero (in-process Python move-gen):")
        print(f"  new wall ≈ {(wall - T.http_t)*1000:.0f} ms  → {wall/(wall - T.http_t):.2f}× faster")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
