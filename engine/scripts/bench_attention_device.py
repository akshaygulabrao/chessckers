#!/usr/bin/env python
"""Benchmark: MPS vs CPU for the V3 transformer attention block.

Times three trunk components at the production dims (c_filters=96, 4 heads,
100 tokens / 10x10 board) on CPU and MPS, in both inference (forward only) and
training (forward+backward) modes, across batch sizes — to answer whether the
attention V3 adds is GPU-friendly enough to revisit the device split. (The
ResNet-only conclusion was: train on MPS, self-play on CPU; see the
project-selfplay-parallelism memory.) The conv ResidualBlock is the baseline the
attention interleaves with; the full V3 trunk (9 res + 7 tf) is the practical
anchor — speedup ratio = cpu_ms / mps_ms (>1 means MPS is faster).

MPS dispatch is asynchronous, so every timed region is bracketed by
torch.mps.synchronize(); each cell is warmed up first (MPS compiles kernels
lazily, so the first calls are unrepresentative).
"""
from __future__ import annotations

import platform
import statistics
import time

import torch

from chessckers_engine.encoding import POS_C_V2
from chessckers_engine.model import ChesskersScorerV2, ResidualBlock, TransformerBlock2d

C = 96          # c_filters (production default)
HEADS = 4
FF_MULT = 4
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
WARMUP = 5
MAX_ITERS = 50
TIME_BUDGET = 1.0  # s/cell cap (bounds slow CPU / large-batch cells); >=5 samples always

WORKLOADS = [
    ("conv", "ResidualBlock (conv baseline)", C),
    ("attn", "TransformerBlock2d (ATTENTION)", C),
    ("trunk", "Full V3 trunk: 9 res + 7 tf", POS_C_V2),
]


def _make_module(kind: str) -> torch.nn.Module:
    if kind == "conv":
        return ResidualBlock(C)
    if kind == "attn":
        return TransformerBlock2d(C, HEADS, FF_MULT)
    if kind == "trunk":
        return ChesskersScorerV2(
            n_blocks=9, n_tf_blocks=7, n_heads=HEADS, tf_ff_mult=FF_MULT
        ).position_trunk
    raise ValueError(kind)


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def _time_cell(module: torch.nn.Module, x: torch.Tensor, train: bool, device: str) -> float:
    """Median per-call latency (ms). Inference = forward under inference_mode;
    train = forward + scalar-loss backward."""
    if train:
        def run() -> None:
            module.zero_grad(set_to_none=True)
            out = module(x)
            out.float().pow(2).mean().backward()
    else:
        def run() -> None:
            with torch.inference_mode():
                module(x)

    for _ in range(WARMUP):
        run()
    _sync(device)

    samples: list[float] = []
    t_start = time.perf_counter()
    for _ in range(MAX_ITERS):
        t0 = time.perf_counter()
        run()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)
        if time.perf_counter() - t_start > TIME_BUDGET and len(samples) >= 5:
            break
    return statistics.median(samples)


def main() -> int:
    torch.manual_seed(0)
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])

    print(f"torch {torch.__version__} | {platform.platform()}")
    print(f"machine: {platform.processor() or platform.machine()} | "
          f"cpu threads={torch.get_num_threads()} | "
          f"mps available={torch.backends.mps.is_available()}")
    print(f"dims: c_filters={C}, heads={HEADS}, ff_mult={FF_MULT}, tokens=100 (10x10)")
    if "mps" not in devices:
        print("\n!! MPS not available — nothing to compare. Exiting.")
        return 1

    for kind, label, c_in in WORKLOADS:
        for train in (False, True):
            mode = "train (fwd+bwd)" if train else "inference (fwd-only)"
            print(f"\n=== {label} | {mode} ===")
            print(f"{'batch':>6}  {'cpu_ms':>9}  {'mps_ms':>9}  {'cpu/mps':>8}  winner")
            crossover: int | None = None
            for B in BATCHES:
                lat: dict[str, float | None] = {}
                for device in devices:
                    try:
                        module = _make_module(kind).to(device)
                        module.train() if train else module.eval()
                        x = torch.randn(B, c_in, 10, 10, device=device)
                        lat[device] = _time_cell(module, x, train, device)
                        del module, x
                        if device == "mps" and hasattr(torch.mps, "empty_cache"):
                            torch.mps.empty_cache()
                    except Exception as e:  # OOM / unsupported op → record, keep sweeping
                        lat[device] = None
                        print(f"{B:>6}  ({device} failed: {type(e).__name__}: {e})")
                cpu_ms, mps_ms = lat.get("cpu"), lat.get("mps")
                if cpu_ms is None or mps_ms is None:
                    continue
                ratio = cpu_ms / mps_ms
                winner = "MPS" if ratio > 1.0 else "CPU"
                if ratio > 1.0 and crossover is None:
                    crossover = B
                print(f"{B:>6}  {cpu_ms:>9.3f}  {mps_ms:>9.3f}  {ratio:>7.2f}x  {winner}")
            if crossover is None:
                print("  crossover: CPU wins at every batch (MPS never faster)")
            else:
                print(f"  crossover: MPS first wins at batch >= {crossover}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
