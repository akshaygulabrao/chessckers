# Run 26 — mate_bench: Gumbel S2 (Sequential Halving @ 64v) vs run-25 control

> Tests Gumbel AlphaZero STAGE 2 — Gumbel-top-m root sampling + Sequential Halving
> visit allocation (Danihelka et al. 2022) — at a 12.5× smaller visit budget
> (`--visits=64` vs 800). Design mirrors run 25's gates-off 2+2 exactly: the two
> **control trials are run 25's arm A** (A1/A2, seeds 0–1: V@90% = **147.2M / 187.2M**,
> gates-off, no PCR, `3688a2a`-era engine) and this run adds only the S2 arm
> (2 trials, seeds 0–1). Metric: search visits to MATE-crossing (`bench_visits.py`,
> window 1000, threshold 0.9, self-play-only).

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `run26_e8d8_gumbelS2_bench` |
| Start FEN | `3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] w - - 0 1` (compiled `board.cc`, unchanged) |
| Arch / trainer | v5 c64/b6, Adam 1e-3, `improved` policy target, pure z, cold, EMA 0.99, publish 400, seeds 0–1 — identical to run-25 arms |
| Self-play params | `--visits=64 --gumbel-sh=true --gumbel-m=16` and NOTHING else (no Dirichlet flags, no temperature flags — Gumbel root perturbation is the exploration mechanism and the SH winner is the played move) |
| Gates | **disabled** (`matches.disabled=true`, carried over from run 25) → league/panel inert, every upload auto-promotes |
| Engine | fork `chessckers-port` + Gumbel S2 commit (on top of `4545d43` slim-edge+memo) |
| Fleet box | vast `44287736` (RTX 3060) |
| Control | run 25 gates-off arm A stamps + tars (`engine/weights/run25-bench-artifacts/`) — NOT re-run |

## Hypothesis / decision rules (pre-committed)

- **H1 (the S2 bet):** SH at n=64 preserves enough policy-improvement signal that
  MATE-crossing arrives at **fewer total search visits** than the 800v control
  (arm A median 167.2M, spread 147–187M). The mechanism: 64v games are ~12.5×
  cheaper per move, so the trainer sees ~an order of magnitude more games/positions
  per visit; Gumbel's improved-policy target is explicitly designed to stay a
  policy improvement at tiny n.
- **Decisive win:** both S2 trials cross below 147M (the control's best seed).
  **Decisive loss:** both above 187M (worst control seed) or a 10h DNF.
  In between: seed-paired reads + games/plies texture, verdict "suggestive, n=2".
- DNF (10h) counts against the arm. Babysitter restarts logged, trial stands.
- Visit accounting: no PCR → every ply is a full search → V = records × 64
  (`bench_visits.py` reads `--visits` from the DB trainParams; verify
  `full_visits: 64, pcr_full_prob: 1.0` in its output before trusting V).
- Caveats accepted upfront: control ran on `3688a2a`, S2 arm runs on the newer
  slim-edge+memo+S2 engine (run 25 verified slim-edge semantically neutral); the
  temperature/noise exploration mechanism differs BY DESIGN (that IS Gumbel S2);
  wall-clock remains tenant-noise-dominated — visits is the metric.

## Design delta (vs run 25 arm A)

Gumbel S2 implemented in the fork (`src/search/classic/`): when `--gumbel-sh` is on,
root child selection ignores PUCT/Dirichlet/temperature. At search start the root
samples g(a)+logP(a) per edge (Gumbel-top-m, m=16 default), then a Sequential
Halving schedule allocates the remaining visit budget (VisitsStopper bound −
initial tree visits, so tree reuse Just Works) in phases — 16×1 → 8×2 → 4×4 → 2×8
at n=64 — re-ranking survivors between phases by g + logP + σ(q̂) with the same
σ convention as the S1 improved-policy target (c_visit=50, c_scale=0.1, min-max q).
The played move is the final SH winner (root exploration = the Gumbel draw itself).
Non-root selection stays PUCT (documented deviation from the paper; subtrees at
n=64 are tiny). Phase advancement waits for in-flight visits to back up (root
collision + batch flush). S1 improved-policy chunk target unchanged. UCI and
default-flag selfplay byte-path unchanged (flag defaults off).

- `params.{h,cc}`: `--gumbel-sh` (bool, default false), `--gumbel-m` (int, default 16)
- `search.{h,cc}`: `SetGumbelVisitBudget`, `GumbelMaybeInit`/`GumbelPickRootChild`/
  `GumbelBestCandidate`, root-branch in `PickNodesToExtendTask`, SH-winner override
  in `EnsureBestMoveKnown`, Dirichlet skipped under gumbel
- `selfplay/game.cc`: budget handoff after Search construction
- `lczero-server` bootstrap: env-driven `VISITS` / `GUMBEL_SH` / `GUMBEL_M` →
  trainParams (S2 arm emits ONLY the three S2 flags)

## Log

- `07-24` Implemented + verified on Mac (Metal): 4-game smoke at 64v — npm 64.4
  (budget consumed exactly), varied openings across games (Gumbel stochasticity
  live), all three outcomes observed; SH winner's root visit share ≈ 0.27–0.28
  (= 15–17/63, textbook schedule). Training chunks decode with **PyVariant oracle
  parity on every record** (192 records / 2 games), improved_policy present +
  normalized. Rules battery green (8000-position parity corpus, rules scenarios,
  FEN invariants — 0 mismatches). UCI default path sane (`bestmove` at 128 nodes,
  gumbel off). Observation for the file: `--temperature=1.0 --tempdecay-moves=15`
  yields 0 games on Mac/Metal at 64v on the PRISTINE `4545d43` binary too —
  pre-existing local quirk (suspect: NDEBUG `assert(sum)` no-op path in
  `GetBestRootChildWithTemperature`), not an S2 regression, not a box blocker
  (run 25 ran those flags for 8.7k games); S2 bypasses temperature entirely.

- `07-24 ~17:46 UTC` Deployed to box `44287736`: fork S2 sources rsynced (box
  fork/server are rsync copies, NOT git — fork lives at
  `/workspace/chessckers/akshay-chessckers-0`), engine target rebuilt (CUDA),
  **box S2 smoke green: 3 games @64v, npm 64.49**. lczero-server bootstrap env
  chain + driver scripts synced; `bench_resume.sh` (single-arm run26, TRIALS=2)
  installed at `/workspace/chessckers/bench_resume.sh`; `*/5` driver cron
  re-armed (was removed at run-25 close); memguard cron still live; gates
  still disabled. Driver self-arms: rewrites `@reboot` to run26 env,
  reset+fresh-launch seed 0.

- `07-24 18:09 UTC` **Trial 1 (seed 0): MATE-crossed at 8,452 games / 18m**
  (window at crossing b=900 w=14 d=86; decisive-only Black 98.5%; 26,761 games/h
  — ~12× the 800v control's throughput, as designed). Zero OOM kills (counter
  pinned at 328), no babysitter restarts. Artifacts archived pre-reset
  (`trial1.db` + 211MB `trial1_games.tar.gz`). Trial 2 (seed 1) auto-launched.
- `07-24 18:15 UTC` **Trial 1 visits (bench_visits, same tool + same dir as the
  control): V@90% = 31.9M** vs control 147.2M / 187.2M (median 167.2M) — a
  **5.2× reduction vs the control median, 4.6× vs its best seed**. Dominates at
  every threshold: V@50% 11.7M (vs 105.0/140.9M), V@75% 31.3M (vs 127.4/175.5M).
  Mechanism, exactly: S2 needed **2.7× MORE plies** (498,283 vs 184,016 — more
  experience) at **12.5× cheaper plies** (64 v/ply vs 800) = 4.6× net. Arithmetic
  check: 498,283 × 64 = 31.89M ✓ (dense accounting, no PCR term).
  Texture: S2 games are SHORTER (59 plies/game vs the control's 79). Honest
  confound to name — S2 removes temperature sampling by design (the played move
  is the SH winner), so some of the ply saving is "no temperature shuffling"
  rather than Sequential Halving per se. It is part of the treatment, not a bug,
  but a 800v-no-temperature control would be the clean way to separate them
  (candidate follow-up, NOT run here).

- `07-24 18:33 UTC` **Trial 2 (seed 1): MATE-crossed at 6,419 games / 23m**
  (window b=900 w=47 d=53). Zero OOM kills all run (counter pinned 328 start to
  finish), no babysitter restarts, no censoring — the cleanest bench yet.

## Result

**DECISIVE WIN — Gumbel S2 at 64 visits crosses on ~5× fewer search visits than
the 800v control, and clears the pre-committed bar (both trials below the
control's BEST seed) with ~4.6× margin to spare.**

| trial | games | plies | V@50% | V@75% | **V@90%** | wall |
|---|---|---|---|---|---|---|
| control A1 (800v, seed 0) | 2,323 | 184,016 | 105.0M | 127.4M | **147.2M** | 1h09m |
| control A2 (800v, seed 1) | 2,933 | 233,955 | 140.9M | 175.5M | **187.2M** | 1h51m |
| S2 T1 (64v, seed 0) | 8,452 | 498,283 | 11.7M | 31.3M | **31.9M** | 18m |
| S2 T2 (64v, seed 1) | 6,419 | 496,852 | 19.7M | 21.7M | **31.8M** | 23m |

- **Arm medians: 31.8M (S2) vs 167.2M (control) = 5.3×.** Seed-paired: seed 0
  4.6×, seed 1 5.9×. Every S2 trial beats every control trial at EVERY
  threshold (50/75/90) — not a 90%-threshold artifact.
- **Mechanism (exact):** S2 needs **2.4× more plies** (~497k vs ~209k median —
  genuinely more experience) at **12.5× cheaper plies** (64 vs 800 v/ply):
  2.4 / 12.5 ⇒ 5.3×. Arithmetic checks: 498,283 × 64 = 31.89M ✓.
- **Variance collapse (suggestive, n=2):** the two S2 trials agree to **0.3%**
  in V (31.8 vs 31.9M) and to **0.3% in plies** (496,852 vs 498,283) despite a
  32% difference in GAME count — i.e. crossing is governed by total position
  experience, not games. The control's two seeds differ by **27%** (147 vs
  187M). If it holds up, S2 is not just faster but far more reproducible —
  which would also mean future A/Bs on this metric need fewer trials.
- **Earlier "S2 games are shorter" note (trial-1 only) is RETRACTED:** T1 was 59
  plies/game but T2 was 77, vs control 79/80. Game length is NOT a robust arm
  effect, which also **weakens the temperature-removal confound** — if losing
  temperature shuffling were driving the win via shorter games, both S2 trials
  would show it. They don't.
- Ops: zero OOM kills across both trials (counter 328 throughout; the slim-edge
  engine holding at a brand-new op-point — 64v, ~27k games/h), no restarts, no
  censoring. Wall-clock 18m/23m vs 1h09m/1h51m (directionally consistent, but
  wall-clock remains tenant-noise-dominated — visits is the verdict).

> **⚠ ABLATION RESOLVED 2026-07-24 — see [run27.md](run27.md). Caveat 1 below is
> now ANSWERED, and it reassigns the credit for this run's win.** PUCT at 64
> visits (the control's root algorithm, control's flags, budget dropped 800→64)
> crosses at **28.1M median — BETTER than S2's 31.8M**. So the 5.3× reported
> here is a **VISIT-BUDGET effect (5.9×), not a Sequential Halving effect**;
> the algorithm contributes no detectable sample-efficiency gain at n=64 on
> this task. What survives for S2 is **wall-clock**: it crossed in 18m/23m vs
> PUCT's 37m/50m by sustaining ~4.4× more visits/hour on the same GPU.
> Read this run's "S2 wins 5.3×" claim as "64 visits wins 5.9×".

**Caveats (what this does NOT establish):**
1. **The ablation is missing.** *(ANSWERED — see the box above and run27.md.)*
   S2 changes TWO things at once vs the control:
   visit budget (800→64) AND root algorithm (PUCT+Dirichlet+temperature →
   Gumbel-top-m + Sequential Halving). This run cannot say whether plain PUCT at
   64v would win too — i.e. whether the credit belongs to "low visits are enough
   on this task" or to "SH is what makes low visits work." The literature's prior
   is the latter (Gumbel exists precisely because PUCT loses the
   policy-improvement guarantee at small n), but prior ≠ measurement.
   **Next experiment: 64v with `--gumbel-sh=false` (2 trials, same seeds).**
2. **One start position, one net size.** e8/d8 KK-vs-K is small and tactical;
   the full start (run-19 monoculture diagnosis) behaves very differently.
   Whether 64v SH holds at high branching + deeper strategy is untested.
3. **n=2**, gates off (league/panel inert), control on `3688a2a` vs S2 on the
   slim-edge+memo+S2 build (run 25 verified slim-edge semantically neutral).
4. Gate/promotion dynamics unmeasured: S2's ~12× game throughput will interact
   with gate cadence and replay-buffer turnover once gating returns.

Artifacts: `engine/weights/run26-bench-artifacts/` (2 trial DBs + game tars +
`bench_visits.json` + `BENCH_RESULTS.jsonl`). Fleet idle; box kept.
`matches.disabled=true` STILL set in box serverconfig — decide gating posture
before the next real run.
