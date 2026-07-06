# Run 15 — Gumbel Stage-1 improved-policy target + pure-z value (full start, attempt 3)

> Run 14 died twice in the frozen-generator distillation spiral — the wm2 search fix (attempt 2)
> was live and verified but did not cure it. Run 15 is the designed attempt-3 remedy: the SAME
> experiment (official full start, c64/b6, warm from run 13) with only the two **training targets**
> changed — policy trains on the Gumbel improved policy instead of raw visit counts, and value
> trains on pure game outcome (`z`) instead of the 0.5 q-mix. First fleet deployment of the
> `feat/gumbel-selfplay` branches (Stage 1: post-search readout only; search behavior unchanged).

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `V5_fullstart_c64b6_gumbelS1` |
| Start FEN | **official full start** (= PyVariant `STARTING_FEN`, identical to runs 10/14; already compiled into fork `board.cc kStartposFen` — no board edit this run) |
| Arch | SE-ResNet gather head, **c_filters=64, n_blocks=6, ~630K params**, tag `v5` (same as runs 11–14) |
| Optimizer | Adam, lr=1e-3 (flat) — same as runs 11–14 |
| **Policy target** | **`improved`** — Gumbel improved policy `softmax(logP + σ(completedQ))`, computed at the root by the fork per recorded ply (`improved_policy` in every ccz record), consumed via `--policy-target improved` (per-example fallback to visits for records lacking the field) |
| **Value target** | **pure z** — `VALUE_Q_RATIO=0` (run 14 used 0.5; the frozen teacher's outcome-decoupled q was half of every value target) |
| Rules | v6 bottom-*d* charge (same as runs 7–14) |
| Init | **WARM-START from run 13's net** (`/workspace/run13_seed/weights.pt` — the same seed as run 14, kept for comparability) |
| Replay buffer | fresh/empty (wiped by `reset_fleet`; window ramp 400→4000g @α0.75) |
| Gate | fresh — 40-game candidate-vs-best, calcElo>−20; first published net bootstrap-promotes |
| Key commits / branch | `feat/gumbel-selfplay` in all three repos — fork `79702a7` (emission; atop wm2 fix `45349d9`), engine `4380def` (trainer consumption) + `a7ab180` (cc plumbing), lczero-server `89ac12e` (bridge plumbing) + `3e3da12` (restart_fleet knobs) |
| Fleet box | vast id `42618148` (RTX 3060), same box as runs 11–14 |
| Started | 2026-07-05 |
| Status | **active** |

## Hypothesis

Run 14's verified failure mechanism: the warm seed is OOD-overconfident on the full start (value
≈ +0.86 everywhere vs ~40/60 actual outcomes) → a blind value function breaks the AZ improvement
operator — search Q's don't separate siblings, so **visit-count policy targets ≈ prior + Dirichlet +
temp** (no improvement signal), while `VALUE_Q_RATIO=0.5` imports the teacher's blindness into the
value target. Run 15 attacks both targets directly:

1. **Improved-policy target** (Gumbel Stage-1): trains the policy toward `softmax(logP + σ(Q̂))`,
   which sharpens toward what search *learned about Q* rather than where noisy visits landed.
2. **Pure-z value**: the value head learns real outcomes only — the bootstrap that un-blinds the
   teacher (completedQ is only informative once value is non-blind).

**Success** = the gate keeps promoting past the run-14 freeze point (nets #4+ promote; candidate-as-
Black win% vs the early best trends up). **Failure signature** = 2–3 early promotions then a
monotone reject wall (−300ish) while trainer metrics look healthy — recognize it early this time.

## Design delta vs run 14

- **Fork `79702a7`** — every ccz record carries `improved_policy` (root readout post-search; wm2-aware
  same-mover flip at `{wm:2}` roots; **no search-behavior change** — moves played are identical in law
  to run 14 attempt 2).
- **Trainer `4380def`** — `train_continuous --policy-target visits|improved` (env
  `CHESSCKERS_POLICY_TARGET`); `_batch_loss` swaps the per-example target with fallback to visits.
- **`VALUE_Q_RATIO` 0.5 → 0** (env-plumbed knob, applied at batch time).
- **Plumbing** — `cc fresh-run --policy-target=/--value-q-ratio=` → `POLICY_TARGET`/`VALUE_Q_RATIO` env
  → `launch_trainer.sh` → bridge → trainer (`a7ab180`, `89ac12e`); persisted in the `@reboot` cron and
  defaulted in `restart_fleet.sh` (`3e3da12`) so a reboot/manual restart can't flip the arm back;
  `cc restart-trainer` now derives the run's knob env from the installed cron instead of hardcoding v4.
- Everything else identical to run 14: start FEN, arch, Adam 1e-3, rules, gate, window ramp,
  parallelism 32, 800-visit self-play / 128-visit gate.
- **Known deviation (watch during A/B):** the fork computes the improved target from **Dirichlet-noised**
  root priors (post-noise `GetP()`), and at 800 visits σ dominates logP → the target is near-one-hot
  (~1/3 of plies argmax a non-top-visit move). `c_scale` (fork, =1) is the softening knob; Stage 2
  (Gumbel search proper) removes the noise at the source.

## Log

- `07-05` **Staged.** Closed run 14 (frozen twice; final net + DB backed up to
  `~/chessckers-backups/run14-fullstart-c64b6-20260705/`). Checked out `feat/gumbel-selfplay` in
  engine + fork (server repo already on it); closed the deploy-plumbing gaps (`a7ab180`, `3e3da12`).
  Launched via `cc fresh-run --run-name=V5_fullstart_c64b6_gumbelS1 --arch=v5 --c-filters=64
  --n-blocks=6 --se-ratio=8 --base=/workspace/run13_seed/weights.pt --policy-target=improved
  --value-q-ratio=0`.
- `07-05` **LIVE + deploy verified end-to-end** (independent 8-check pass): trainer spawned with
  `--policy-target improved --value-q-ratio 0.0`, warm-loaded the run-13 seed at c64/b6; DB
  bootstrapped `training_run #1 "V5_fullstart_c64b6_gumbelS1"`, first net bootstrap-promoted; fork
  binary rebuilt at launch (19:37 UTC). First production chunk decoded locally with the branch
  trainer code: `improved_policy` on **21/21 plies** (normalized, aligned with visits),
  `wdl_target=[0,0,1]` **pure z**, **q(ply0)=+0.84 ≈ q(ply1)=+0.87 same-sign** (wm2 fix intact),
  official start FEN with `{wm:2}`. Games dir created fresh at launch → no attempt-2 chunk leakage.
  GPU ~73%, 13 chunks in the first ~10 min; `@reboot` cron carries
  `POLICY_TARGET=improved VALUE_Q_RATIO=0`.

- `07-06` **Gate stall diagnosed + threshold soak.** Best froze at #5 (21:46) — #6–#9 rejected at
  −108/−53/−70/−89 (shallow, non-monotonic; cf. run 14 at the same point: −108/−158/−147/−108→−241).
  Forensics: candidate-as-Black flat (9→13→12→12), vsign ~0.87, |root_q| median 0.86→0.68 (seed
  optimism decaying under pure z), improved-target argmax agreement with visits RISING 46%→63% —
  the improvement operator works; the blocker is the gate fixed point (the postmortem's un-pulled
  remedy). NOT the run-14 slide. Note: root_q-vs-z sign agreement 74%→41% is the outcome
  distribution shifting Black-ward (White wins 67%→25% in sampled cohorts) under a lagging,
  still-White-optimistic value head — calibration lag, not a new sign bug (wm2 q0≈q1 100%).
  **Intervention (00:52 UTC):** box-side `serverconfig.json matches.threshold` −20 → **−100** +
  state-preserving server restart (soak; local config stays −20 so future runs revert). Revert to
  −20 once promotions flow and cumElo rises (or two consecutive candidates score ≥ −20). Tripwire
  by design: a real slide (< −100) still freezes the gate. First post-soak decision: #10 at −127
  (below the soak bar → still reject; next matches decide stall-vs-slide). `cc status`/`cc strength`
  now display the run identity (this session's display patch).

## Result

<active — leave empty. Primary read: does the gate keep promoting past the ~#4 freeze point, and does
candidate-as-Black win% vs the early best trend up? Link successor run when pivoted.>
