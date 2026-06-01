#!/usr/bin/env python
"""NN throughput ceiling: model.batch_eval positions/sec vs batch size + device.

This isolates the pure network cost that MCTS pays per leaf — trunk + value
head + (ragged) policy head — over a realistic pool of (position, legal_moves)
pairs harvested by random rollouts. Encoding is done ONCE up front (excluded
from timing); the timed loop measures stack -> device-transfer -> batch_eval ->
read values back to CPU, which is exactly what a vectorized self-play step pays.

Run one device per invocation (clean isolation — no CPU/GPU contention, and
MPS shader-compile warmup doesn't pollute CPU timing):

  .venv/bin/python bench/bench_nn_throughput.py --device cpu  --threads 1
  .venv/bin/python bench/bench_nn_throughput.py --device cpu  --threads 8
  .venv/bin/python bench/bench_nn_throughput.py --device mps

The decisive number is positions/sec. Compare MPS@large-B against
(CPU@B=1,threads=1) x n_cores — that product is what the current N-process
fleet delivers. MPS is only worth chasing if it clears that aggregate.
"""
from __future__ import annotations

import argparse
import random
import statistics
import time

import torch

from chessckers_engine.encoding import encode_move, encode_position_state
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.variant_py import PyVariantClient

# Mix of small (endgame) and large (full-board) move lists so the ragged
# padding in batch_eval is exercised realistically.
SEEDS = [
    "7p/8/8/8/8/8/8/4K3[h8:ssss] b - - 0 1",
    "8/8/3kkk2/8/8/8/3PPP2/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
    "8/8/8/8/3kk3/8/8/4K3[d4:kk,e4:kk] b - - 0 1",
    "pppppppp/kkkkkkkk/1ppppppp/1p6/3PP3/8/PPP2PPP/RNBQKBNR"
    "[b5:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,a7:k,b7:k,c7:k,d7:k,e7:k,"
    "f7:k,g7:k,h7:k,a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] b KQ - 0 1",
]


def harvest_pool(n: int, rng: random.Random) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Random-rollout a pool of (encoded_pos (C,8,8), encoded_moves (N,D)) on CPU."""
    client = PyVariantClient()
    pool: list[tuple[torch.Tensor, torch.Tensor]] = []
    state = client.parse(rng.choice(SEEDS))
    depth = 0
    while len(pool) < n:
        status, _winner, legal = client.status_and_legal(state)
        if status or not legal or depth > 60:
            state = client.parse(rng.choice(SEEDS))
            depth = 0
            continue
        pos = encode_position_state(state)
        moves = torch.stack([encode_move(m) for m in legal])
        pool.append((pos, moves))
        state = client.apply_known(state, rng.choice(legal))
        depth += 1
    return pool


def time_batch(
    model: ChesskersScorer, pool: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device, B: int, n_batches: int, rng: random.Random,
) -> float:
    """Return seconds for n_batches batched evals of size B (transfer+compute+readback)."""
    is_mps = device.type == "mps"
    for _ in range(n_batches):
        idx = [rng.randrange(len(pool)) for _ in range(B)]
        positions = torch.stack([pool[i][0] for i in idx]).to(device)
        moves_list = [pool[i][1].to(device) for i in idx]
        with torch.no_grad():
            values, _priors = model.batch_eval(positions, moves_list)
        _ = values.tolist()  # forces device->CPU sync (realistic readback)
    if is_mps:
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_batches):
        idx = [rng.randrange(len(pool)) for _ in range(B)]
        positions = torch.stack([pool[i][0] for i in idx]).to(device)
        moves_list = [pool[i][1].to(device) for i in idx]
        with torch.no_grad():
            values, _priors = model.batch_eval(positions, moves_list)
        _ = values.tolist()
    if is_mps:
        torch.mps.synchronize()
    return time.perf_counter() - t0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", help="cpu|mps")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = leave default)")
    p.add_argument("--batches", type=int, default=80, help="timed batched evals per B")
    p.add_argument("--pool", type=int, default=384)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--weights", default="")
    p.add_argument("--d-hidden", type=int, default=256)
    p.add_argument("--c-filters", type=int, default=96)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resident", action="store_true",
                   help="pre-move pool to device (compute-only; no per-call transfer)")
    args = p.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS not available on this machine")
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    rng = random.Random(args.seed)

    model = ChesskersScorer(
        d_hidden=args.d_hidden, c_filters=args.c_filters, n_blocks=args.n_blocks,
    ).to(device).eval()
    if args.weights:
        from chessckers_engine.checkpoints import load_checkpoint
        load_checkpoint(model, args.weights)

    pool = harvest_pool(args.pool, rng)
    avg_moves = statistics.mean(m.shape[0] for _, m in pool)
    if args.resident:
        # Pre-move the pool to device: the per-call .to(device) becomes a no-op
        # (already-on-device tensors return self), so the timed loop measures
        # pure compute + on-device stack + value readback — no host->device
        # transfer. Bounds the optimization headroom over the realistic path.
        pool = [(p.to(device), m.to(device)) for p, m in pool]

    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256] if device.type == "mps" else [1, 8, 32, 64, 128]

    print(f"=== NN throughput: device={args.device} threads={torch.get_num_threads()} "
          f"net={args.d_hidden}/{args.c_filters}/{args.n_blocks} ===")
    print(f"pool={len(pool)} positions, avg {avg_moves:.1f} legal moves/pos, "
          f"{args.batches} batches x {args.runs} runs each\n")
    print(f"{'B':>5} {'ms/batch':>10} {'pos/sec':>12} {'speedup/B1':>11}")
    print("-" * 42)
    base_pos_per_sec = None
    for B in batch_sizes:
        secs = [time_batch(model, pool, device, B, args.batches, rng) for _ in range(args.runs)]
        best = min(secs)  # best-of-runs = least noise
        pos_per_sec = B * args.batches / best
        ms_per_batch = best / args.batches * 1000
        if B == 1:
            base_pos_per_sec = pos_per_sec
        spd = pos_per_sec / base_pos_per_sec if base_pos_per_sec else 1.0
        print(f"{B:>5} {ms_per_batch:>10.3f} {pos_per_sec:>12.0f} {spd:>10.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
