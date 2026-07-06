# Run 17 — cold start, no gate (classic AZ loop on the full start)

> User-directed pivot after the run-14/15 postmortems implicated two structural causes: the
> **warm-start transplant** (confidently-wrong value head from a different position) and the **gate
> fixed point** (frozen generator → own-equilibrium data → no improvement signal). Run 17 removes
> both at once: **random init** (no seed at all) and **promote-always** (gate threshold −9999 — every
> candidate promotes; the 40-game matches still run and are recorded, turning the gate into a pure
> strength-measurement series). This is the closest this fleet has come to lc0/AZ's actual operating
> point: always-latest generator, soft visit targets, pure-z value.

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `V5_fullstart_c64b6_cold_nogate` |
| Start FEN | official full start (= `STARTING_FEN`, same as runs 10/14/15/16) |
| Arch | SE-ResNet gather head, c64/b6 ~630K, tag `v5` |
| Optimizer | Adam, lr=1e-3 |
| Policy target | `visits` (classic AZ; Gumbel code physically absent — same pre-Gumbel trees as run 16) |
| Value target | pure z (`VALUE_Q_RATIO=0`) |
| **Init** | **COLD random init** (no `--base`; deterministic seed-0). First cold full-start run since the curriculum began — expectations set accordingly (see Hypothesis). |
| **Gate** | **REMOVED** — `serverconfig.json matches.threshold = -9999` ⇒ promote-always (calcElo bottoms out ≈ −800, so every match passes). Matches (40g @128v) still played + recorded as a pure measurement series; `cc strength` cumElo becomes the run's strength trajectory, not a filter. |
| Rules | v6 bottom-*d* charge |
| Replay buffer | unchanged (window ramp 400→4000 @α0.75, RF=8) — buffer redesign (mixed sampling / outcome cap / target averaging / RF cut) **deliberately deferred** so the two changes stay readable |
| Key branches | same `ctl/pre-gumbel-run16` control trees (fork `45349d9`, engine `5615196`+docs, server `dcbe1df` + this gate-config commit) |
| Fleet box | vast `42618148` (RTX 3060) |
| Started | 2026-07-06 |
| Status | **active** |

## Hypothesis

With the generator never frozen (promote-always ⇒ effectively play-latest with ~one-publish lag) and
no transplanted value head, does the classic AZ loop bootstrap on the full start?

- **Success read:** cumElo over the promote-always series trends up over many nets; game quality
  visibly improves (`cc games`); balance settles somewhere decisive without value-head whiplash.
- **Failure read (new signature — no reject wall exists anymore):** *sustained* cumElo decline across
  many promotions = drift without gate protection; or flatline ≈ 0 for a long horizon = cold start too
  slow at this net size / single-start value starvation (the remaining un-fixed cause).
- **Expectations:** cold on the full game is the thing the runs 5→13 curriculum existed to avoid —
  early progress may be slow (hours–days before coherent play). The per-match Elo is now pure
  measurement; single matches are ±110 noisy, judge the *cumulative* trend.
- **What this run does NOT test:** the Gumbel-bug question (run 16's cell, aborted pre-verdict,
  reproducible) and the buffer redesign (deferred to run 18 candidates).

## Design delta vs run 16

- Init: warm run-13 seed → **cold random**.
- Gate: −20 filter → **promote-always** (−9999), matches retained as measurement.
- Everything else identical (code trees, target, q-ratio, arch, start, buffer config, parallelism).

## Log

- `07-06` **Staged + launched.** Run 16 aborted pre-verdict (95 games, 0 matches — nothing lost;
  reproducible). Gate disabled via local `serverconfig.json` threshold −9999 (main branch keeps −20 —
  a future non-experiment run reverts by rsync). Launched via `cc fresh-run
  --run-name=V5_fullstart_c64b6_cold_nogate --arch=v5 --c-filters=64 --n-blocks=6 --se-ratio=8
  --value-q-ratio=0` (no `--base` ⇒ cold). Known benign: a cold deterministic init can trip the
  upload SHA-dedup 400 once ("Network already exists") — non-fatal.

## Result

<active — leave empty. Primary read: cumElo trend of the promote-always series over ≥15–20 nets +
game-quality probes. Link successor when pivoted.>
