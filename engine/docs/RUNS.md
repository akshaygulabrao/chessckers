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
| 11 | 2026-06-30→07-01 | **e8/d8 KK-vs-K** `3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] b` (same seed as runs 5–8) | **v5 c64/b6 ~630K** (scaled up from c48/b5 ~364K) | Adam lr=1e-3 (same) | **done** — clean **c64/b6 scale-up baseline on e8/d8**; converged ~99% Black @ ~81.8k games (matches c48/b5 runs 5–8). First step toward competent play → run 12. | [run11.md](runs/run11.md) |
| 12 | 2026-07-01→07-02 | **d6/e6/f6 pawn-wall** `8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] w` (same as run 9) | v5 c64/b6 ~630K (from run 11) | Adam lr=1e-3 (same) | **done** — **Black wins by force** at c64/b6 (~96% Black, flat; human couldn't hold White either), confirming run 9's read. Warm-started from run 11. → run 13. | [run12.md](runs/run12.md) |
| 13 | 2026-07-02 | **full White army vs d6/e6/f6 KK** `8/8/3kkk2/8/8/8/PPPPPPPP/RNBQKBNR[d6:kk,e6:kk,f6:kk] w KQkq - 0 1 {wm:2}` (run-12 towers + full FIDE White + double-move) | v5 c64/b6 ~630K (from run 12) | Adam lr=1e-3 (same) | **done** — **a full White army breaks run 12's Black-forced win**: self-play inverted to **White 91% / Black 9%** (~5.7k games). Warm-started from run 12. → run 14. | [run13.md](runs/run13.md) |
| 14 | 2026-07-02→07-05 | **official full starting position** `pppppppp/pkkkkkkp/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR[…24-tower…] w KQkq - 0 1 {wm:2}` (= `STARTING_FEN`, same as run 10) | v5 c64/b6 ~630K (from run 13) | Adam lr=1e-3 (same) | **dead** — froze **twice** in the frozen-generator distillation spiral (attempt 1 at #4; attempt 2 *with the wm2 search fix* at #3, 41 straight rejects): blind warm seed → visit targets ≈ prior+noise + q-ratio 0.5 value poison. → run 15 | [run14.md](runs/run14.md) |
| 15 | 2026-07-05→07-06 | official full start (same as 14) | v5 c64/b6 ~630K (warm from run 13 — same seed as 14) | Adam lr=1e-3; **policy target = Gumbel improved** (Stage 1), **value = pure z** (q-ratio 0) | **halted** — first net-positive promotions on the full start (+44/+53), then gate froze at #5 and `c_scale=1`'s one-hot target crowned Q-noise → students distilled below the teacher (−98 @v1 → −301 @gate, Black-side collapse). Run 16 needs: c_scale 0.1, freeze protection, opening diversity | [run15.md](runs/run15.md) |

**Deferred ideas** (not numbered): [low self-play visits 800→100](runs/deferred-low-visits.md) —
dropped 2026-06-26 (judged low-impact).

> **Numbering note:** run identity is the triple **(`RUN_NAME` env, compiled-in start FEN,
> arch-version tag)** + optimizer. The 06-15→06-16 e8/d8 ↔ d6/e6/f6 reverts were churn
> *within* the campaign, not separate numbered runs — collapse or split rows to match how
> you actually count runs. Earlier runs (1–4) predate this ledger and aren't backfilled.
