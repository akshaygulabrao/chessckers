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
| Fleet box | vast `44017141` (RTX 3060, KR) — box 2; the original launch box `42618148` went away same-day pre-games |
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
- `07-06` **Box replaced → relaunched on vast `44017141`** (original box `42618148` gone before any
  games). Same `cc fresh-run` command, same trees. Two durable fixes landed in
  `lczero-server/scripts/provision_server_vast.sh` while re-provisioning: (1) engine rsync now
  `--exclude 'weights/'` (1.0 GB of Mac-side A/B nets was silently shipped every provision); (2) the
  engine venv is no longer `uv sync` — the lock resolves PyPI torch cu13x wheels (CPU-only on this
  12.8-driver host) and uv's download also wedged outright on this box; the venv is now
  `--system-site-packages` over the template's `/venv/main` torch (2.11.0+cu128, cuda=True verified)
  + pip for the four small pure-python deps. Run-13 seed also shipped to
  `/workspace/run13_seed/weights.pt` so the `cc anchor` seed13 anchor resolves.
- `07-06` **LIVE + verified** (11:31 box time): server/bridge/trainer UP; DB run #1
  `V5_fullstart_c64b6_cold_nogate`; box gate thr **−9999** confirmed; trainer argv `--arch-version v5
  --c-filters 64 --n-blocks 6 --se-ratio 8 --value-q-ratio 0.0` with **no `--base`** (cold); 0
  `improved_policy` hits in box fork src + engine trainer; cold net #1 uploaded + bootstrap-promoted
  (no dedup 400 — fresh DB); client engines up with CUDA trunk. First chunk verified: raw ccz1 JSON
  has **0 `improved_policy`**, start FEN = official full start `{wm:2}`, value targets pure ±1 z, and
  wm2 same-mover sign intact (ply0 q=−0.028 = ply1 q=−0.028, flip at ply2) — cold-net q ≈ 0 as
  expected. First game: Black win in 23 plies.

## Result

<active — leave empty. Primary read: cumElo trend of the promote-always series over ≥15–20 nets +
game-quality probes. Link successor when pivoted.>
