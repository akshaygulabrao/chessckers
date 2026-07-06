# Run 16 — strict pre-Gumbel control (visits target, pure-z value)

> Falsification run requested after run 15's collapse: "make sure we didn't just introduce a bug"
> with the Gumbel port. All Gumbel code is **physically absent** (strict code rewind, not a config
> flip), and the run fills the missing cell of the 2×2 target matrix:
>
> | | q-ratio 0.5 | q-ratio 0 |
> |---|---|---|
> | **visits** | run 14 att 2 — froze #3, −241 by 6 rejects, −203 @v1 | **run 16 (this run)** |
> | **improved** | — | run 15 — froze #5 after +44/+53, −147 by 6 rejects, −98 @v1 |

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `V5_fullstart_c64b6_visits_qz0` |
| Start FEN | official full start (= `STARTING_FEN`, same as runs 10/14/15) |
| Arch | SE-ResNet gather head, c64/b6 ~630K, tag `v5` (same as 11–15) |
| Optimizer | Adam, lr=1e-3 (same) |
| **Policy target** | **`visits`** (classic AZ visit distribution) — Gumbel emission/consumption code physically absent |
| **Value target** | **pure z** (`VALUE_Q_RATIO=0`) — the only knob carried over from run 15 |
| Rules | v6 bottom-*d* charge |
| Init | warm from run-13 seed (`/workspace/run13_seed/weights.pt`, same seed as 14/15) |
| Replay buffer | fresh (window ramp 400→4000g @α0.75, same) |
| Gate | −20 / 40 games (fresh-run re-rsynced the local serverconfig, reverting run-15's −100 box-side soak) |
| Key branches | `ctl/pre-gumbel-run16` in all three repos: fork @ `45349d9` (wm2 fix tip, pre-Gumbel), engine `5615196` (trainer/analysis rewound to `635c871`; cc/ops/display tooling kept), server `dcbe1df` (bridge + launch_trainer rewound to `0869a18`; restart_fleet/fleet_status kept — its `POLICY_TARGET` env is inert against the pre-Gumbel launch script, verified) |
| Fleet box | vast `42618148` (RTX 3060) |
| Started | 2026-07-06 |
| Status | **aborted pre-verdict → superseded by [run 17](run17.md)** |

## Hypothesis

If the freeze-and-slide recurs with the Gumbel code physically absent, the spiral is **structural**
(the gate fixed point + single-start value starvation), and the Gumbel port is exonerated as a bug
source. Expected readouts:

- **(a) freeze + slide recurs** → structural; compare slide *rate* at matched reject counts:
  run 14 (q-poison arm) −241 by 6 rejects; run 15 (one-hot-target arm) −147 by 6. Run 16's rate
  attributes the damage: similar to 15 ⇒ slide needs no Gumbel target at all; much slower/flat ⇒
  the one-hot target was run 15's accelerant.
- **(b) plateau at replica-parity, occasional promotions** → consistent with the c_scale diagnosis
  (soft visits targets degrade toward the prior, not below the teacher); the freeze remains the
  structural disease.
- **(c) early phase beats run 15's** (+44/+53 net-positive promotions) → the improved target added
  no value even while the value head discriminated — revisit the Stage-1 design entirely.

## Design delta vs run 15

- Policy target improved → **visits**; Gumbel emission (fork), consumption (trainer), and analysis
  hooks **removed from the working trees** (strict rewind), not just disabled.
- Everything else identical: seed, arch, optimizer, start FEN, gate (−20), window ramp, parallelism,
  800-visit self-play / 128-visit gate, pure-z value.

## Log

- `07-06` **Staged + launched.** Control branches cut and verified (0 `improved_policy` hits in fork
  src and engine trainer; `--value-q-ratio` confirmed pre-existing through every hop: cc.py env →
  launch_trainer → bridge → train_continuous). Launched via `cc fresh-run
  --run-name=V5_fullstart_c64b6_visits_qz0 --arch=v5 --c-filters=64 --n-blocks=6 --se-ratio=8
  --base=/workspace/run13_seed/weights.pt --value-q-ratio=0` (no `--policy-target` — the flag's
  consumer doesn't exist in this tree).
- `07-06` **LIVE + control-purity verified** (inverted 7-check pass, 05:20 UTC): zero `improved_policy`
  hits in box fork src / trainer stack / bridge / launcher; **raw chunk JSON contains 0 occurrences of
  `improved_policy`** (the pre-Gumbel binary cannot emit it); trainer argv `--value-q-ratio 0.0` with
  no policy-target, warm-loaded run-13 seed at c64/b6; DB bootstrapped `training_run #1
  "V5_fullstart_c64b6_visits_qz0"`; gate `thr -20` restored (−100 soak gone with the re-rsynced
  config); q(ply0)=+0.84 ≈ q(ply1)=+0.85 same-sign (wm2 fix intact at `45349d9`). Watch: early client
  rate read low (~1.7K games/day vs run-15's ~4.9K) — likely startup transient, recheck at the
  first gate match.

## Result

**Aborted 2026-07-06 before any gate match completed** (95 games, 1 net, step 48 — no verdict data;
nothing worth backing up beyond the already-backed-up seed). Superseded by [run 17](run17.md) (user
pivot: cold init + gate removed, attacking the two suspected root causes directly). The question this
control was built for — does the freeze/slide recur under (warm seed, gate, visits, q=0) with zero
Gumbel code — **remains open** and is fully reproducible: branches `ctl/pre-gumbel-run16` (all three
repos) + the launch command in the Log above.
