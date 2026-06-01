# Batched / vectorized MCTS on MPS — findings (NEGATIVE)

**Date:** 2026-06-01  **Net:** `ChesskersScorer(d_hidden=256, c_filters=96, n_blocks=4)` ≈ 2.48M params
**Machine:** Apple Silicon, `hw.ncpu=8`, 6 performance cores, PyTorch MPS/CPU.

## Question

Self-play currently runs as N independent CPU processes, each doing batch-1 MCTS.
Can we get more throughput by using the GPU (MPS) via batched NN evaluation —
either batching leaves within one tree (Leela-style) or interleaving K games and
batching one leaf per game? And specifically: **if the batch is large enough, does
it make up for the net being tiny?**

## TL;DR verdict

**No, not for this net.** Keep the CPU process fleet. A large batch *does* let MPS
beat the fleet on the **NN forward in isolation**, but (a) host→device transfer
halves that in practice, and (b) the forward is **not** the end-to-end bottleneck
once it's batched — the **serial CPU move-gen/expansion** is, and a single-thread
vectorized engine can't parallelize that across cores the way the fleet does.
Measured end-to-end, the MPS vectorized engine is **~10× slower** than the fleet.

The approach flips from loss to win only with a **much larger net** (the GPU's margin
over CPU grows with net size), or a **parallel-move-gen GPU inference server**
(contingent on the Rust move-gen releasing the GIL).

## Numbers

Baseline to beat — production fleet ≈ 6 × (CPU batch-1, threads=1) ≈ **~5,000 pos/sec**.

### NN throughput, `model.batch_eval` (pos/sec), `bench/bench_nn_throughput.py`

| B | CPU t=1 | CPU t=6 | MPS realistic | MPS compute-only (`--resident`) |
|---:|---:|---:|---:|---:|
| 1 | 826 | 611 | 159 | 196 |
| 8 | 1118 | 1633 | 953 | 1626 |
| 32 | 769 | 1049 | 1928 | 5356 |
| 64 | 800 | 1430 | 2593 | 6124 |
| **128** | 829 | **1829** | **2992** | **6415** |
| 256 | — | — | 2941 ↓ | 5473 ↓ |

- Single-thread CPU does **not** batch (B=1 ≈ B=128 ≈ 800) — one core is compute-bound.
  This is *why* the fleet uses N independent processes.
- One 6-thread batched CPU process peaks ~1,829 — far below the 6-process aggregate.
- MPS **compute-only** peaks **6,415** (> fleet), but the **realistic** path (per-step
  stack + host→device transfer + value readback — what self-play actually pays) peaks
  **2,992** (< fleet). The ~2× gap is transfer/marshaling overhead.
- Both MPS curves **plateau at B=128 and regress at B=256** — batching hits a ceiling.

### End-to-end self-play (400 sims), `bench/vmcts.py`

| config | moves/sec | pos_eval/sec |
|---|---:|---:|
| 1 CPU process (batch-1) | 1.59 | 637 |
| **6-proc fleet** (×6) | **~9.5** | **~3,800** |
| **MPS vectorized, K=64** | **0.94** | **377** |

The vectorized engine is correct (verified bit-for-bit equal to `run_mcts` at K=1,
no noise) but ~10× slower than the fleet end-to-end: it batches the forward
(~0.39 ms/slot) yet serializes all K games' selection/move-gen on **one** core, plus
per-step transfer marshaling — ~1.9 ms/slot of overhead the eval-only numbers hide.

## Why a big batch can't rescue a small net

A forward pass is a chain of many **small, sequential** kernels (conv→GN→relu × depth).
Each kernel launch has fixed latency; for a small net the kernels are too small to
saturate the GPU even at B=128, so you're **launch/occupancy-bound, not compute-bound**.
Batching amortizes per-call *overhead* (climbs you up the curve) but cannot raise the
*arithmetic intensity* — so throughput plateaus at a ceiling and then regresses.

**Net *size* sets the ceiling; batch only climbs to it.** A bigger net has the *same*
number of kernel launches but each does far more work → compute-bound → near peak FLOPS
→ a much higher ceiling, reached at a *smaller* batch. So the GPU's margin over CPU
**grows with net size** and shrinks toward our tiny-net regime.

## When this flips to a win

1. **Bigger net** (the real lever — also the strength lever). At ~10–30M params
   (Leela territory) the forward dominates so heavily that the serial move-gen tax
   becomes noise and batched-GPU wins decisively.
2. **Parallel-move-gen GPU inference server**: N CPU workers do selection/move-gen
   in parallel and submit leaves to one batched GPU evaluator. Only worth it if the
   Rust move-gen **releases the GIL** (otherwise GIL-serialized — which is likely why
   the old thread-based `InferenceServer` was net-negative). **Unverified; check first.**

## Relation to Leela / Maia

- **Leela (Lc0)** is the same AlphaZero family; its heavy intra-tree batching assumes a
  large net where the forward dominates. We're below that size, so process-parallel CPU
  wins. (Lc0's moves-left + WDL value heads are the principled fix for our shortest-mate
  gap — a more valuable borrow than this throughput work.)
- **Maia** is supervised imitation of human moves, policy-only, ~no search — inapplicable
  (no human Chessckers data). Transferable idea: a strong policy can play with few/no sims.

## Reproduce

```
# NN ceiling (run one device per invocation; isolated, no contention):
.venv/bin/python bench/bench_nn_throughput.py --device cpu --threads 1
.venv/bin/python bench/bench_nn_throughput.py --device cpu --threads 6
.venv/bin/python bench/bench_nn_throughput.py --device mps              # realistic
.venv/bin/python bench/bench_nn_throughput.py --device mps --resident   # compute-only

# Correctness + end-to-end:
.venv/bin/python bench/vmcts.py --mode verify
.venv/bin/python bench/vmcts.py --mode cpu-seq --threads 1 --sims 400 --moves 64
.venv/bin/python bench/vmcts.py --mode mps-vec --K 64 --sims 400 --moves 192
```

Artifacts: `bench/bench_nn_throughput.py` (NN-ceiling microbench),
`bench/vmcts.py` (single-thread vectorized PUCT over K games + the harness).
