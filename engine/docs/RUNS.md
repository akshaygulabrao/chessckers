# Training runs — index

One row per training run. Each links to its full doc in [`runs/`](runs/). To start a
run: copy [`runs/_TEMPLATE.md`](runs/_TEMPLATE.md) → `runs/runN.md`, fill the Identity
table from `cc fresh-run` stdout, write a Hypothesis, then keep the Log up to date.

Shared, run-independent arch/encoding tables live in
[`encoding-reference.md`](encoding-reference.md).

| Run | Dates | Start FEN | Arch | Optimizer | Status | Doc |
|---|---|---|---|---|---|---|
| 5 | ~2026-06 | `3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] w` (e8/d8 KK-vs-K) | SE-ResNet c48/b5 v5 ~364K | Adam lr=2e-2 (accidental) | superseded → 6 | [run5.md](runs/run5.md) |
| 6 | 2026-06-25 | `3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] w` (same as run 5) | SE-ResNet c48/b5 v5 ~364K (unchanged) | Adam lr=1e-3 | **done** — converged ~8k games (vs run 5 ~38k) | [run6.md](runs/run6.md) |
| 7 | 2026-06-26 | e8/d8 (from run 6) | v5 c48/b5 (from run 6) | Adam lr=1e-3 (from run 6) | **in impl** — rule change: charge demotes **bottom-N** Kings (no choice) | [run7.md](runs/run7.md) |

**Deferred ideas** (not numbered): [low self-play visits 800→100](runs/deferred-low-visits.md) —
dropped 2026-06-26 (judged low-impact).

> **Numbering note:** run identity is the triple **(`RUN_NAME` env, compiled-in start FEN,
> arch-version tag)** + optimizer. The 06-15→06-16 e8/d8 ↔ d6/e6/f6 reverts were churn
> *within* the campaign, not separate numbered runs — collapse or split rows to match how
> you actually count runs. Earlier runs (1–4) predate this ledger and aren't backfilled.
