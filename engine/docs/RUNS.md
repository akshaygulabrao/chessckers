# Training runs — index

One row per training run. Each links to its full doc in [`runs/`](runs/). To start a
run: copy [`runs/_TEMPLATE.md`](runs/_TEMPLATE.md) → `runs/runN.md`, fill the Identity
table from `cc fresh-run` stdout, write a Hypothesis, then keep the Log up to date.

Shared, run-independent arch/encoding tables live in
[`encoding-reference.md`](encoding-reference.md).

| Run | Dates | Start FEN | Arch | Optimizer | Status | Doc |
|---|---|---|---|---|---|---|
| 5 | ~2026-06 | `3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] w` (e8/d8 KK-vs-K) | SE-ResNet c48/b5 v5 ~364K | Adam lr=2e-2 (accidental) | superseded → 6 | [run5.md](runs/run5.md) |
| 6 | 2026-06-25 | `3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] w` (same as run 5) | SE-ResNet c48/b5 v5 ~364K (unchanged) | Adam lr=1e-3 | **done** — converged ~8k games (vs run 5 ~10k) | [run6.md](runs/run6.md) |
| 7 | 2026-06-26 | e8/d8 (from run 6) | v5 c48/b5 (from run 6) | Adam lr=1e-3 (from run 6) | **done** — bottom-N charge rule; converged ~99.3% Black (games-to-converge vs run 6 TBD) | [run7.md](runs/run7.md) |
| 8 | 2026-06-28 | e8/d8 (same as run 7) | v5 c48/b5 (same) | Adam lr=1e-3 (same) | **done** — infra validation: gate + `cc strength` validated; stopped ~91% Black @ 9.3k games | [run8.md](runs/run8.md) |

| 9 | 2026-06-29 | `8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] w` (d6/e6/f6 vs pawn wall) | v5 c48/b5 (same) | Adam lr=1e-3 (same) | **done** — unknown-answer resolved: **Black wins** (~99% Black @ 16k games); warm-start from run 8 transferred | [run9.md](runs/run9.md) |
| 10 | 2026-06-29→30 | `…/RNBQKBNR[…24-tower…] w KQkq - 0 1 {wm:2}` (**official full starting position**, opening double-move) | v5 c48/b5 (same) | Adam lr=1e-3 (same) | **done** — first run on the complete game; **warm-started from run 9 net**; net swept its run-9 seed **10–0** from the full start (learned a coherent opening; absolute strength TBD) | [run10.md](runs/run10.md) |

**Deferred ideas** (not numbered): [low self-play visits 800→100](runs/deferred-low-visits.md) —
dropped 2026-06-26 (judged low-impact).

> **Numbering note:** run identity is the triple **(`RUN_NAME` env, compiled-in start FEN,
> arch-version tag)** + optimizer. The 06-15→06-16 e8/d8 ↔ d6/e6/f6 reverts were churn
> *within* the campaign, not separate numbered runs — collapse or split rows to match how
> you actually count runs. Earlier runs (1–4) predate this ledger and aren't backfilled.
