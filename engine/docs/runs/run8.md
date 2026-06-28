# Run 8 — infra validation (in-fleet gate + cc strength)

> **Plumbing run, not a science run.** Same training config as [run 7](run7.md) (e8/d8, v6
> bottom-*d* charge rule); the *only* change is **monitoring/promotion infrastructure**. Purpose:
> exercise the re-enabled in-fleet gate and the `cc strength` table end-to-end before using them
> on a real (unknown-answer) experiment.

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `V6_e8d8_gated` |
| Start FEN | `3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] w - - 0 1` (e8/d8, same as run 7) |
| Arch | SE-ResNet v5 c48/b5 ~364K (same as run 7) |
| Optimizer | Adam lr=1e-3 (same) |
| Rules | v6 bottom-*d* charge (same as run 7) |
| **New: gate** | in-fleet lc0 gate — `calcElo > −20`, 40-game candidate-vs-best match per publish |
| Fleet box | vast 42618148 (RTX 3060) — cold-launched 2026-06-28 |
| Status | **active** — infra validated; running as a soak test |

## Purpose (not a hypothesis)

Validate the two pieces built this session against a live fleet:
1. **In-fleet promotion gate** — `uploadNetwork` queues a candidate-vs-best match instead of
   auto-promoting; promote iff `calcElo > Matches.Threshold` (−20, lc0-lenient).
2. **`cc strength`** — table-only net-vs-past-selves check (`gauntlet --no-curve`) + `--out` history.

## Design delta vs run 7 (infra only — no training change)

- **Server (`lczero-server` `9a4f7f0`):** `uploadNetwork` → bootstrap-promote the first net, else
  `createMatch` (target_slice 0 so a 1-node fleet plays it). `checkMatchFinished` promotes on
  `calcElo > −20` and now logs a `[gate]` promote/reject line. `serverconfig` `Matches.Games 40→8`
  (slice-0 ×5 ⇒ a 40-game match, not 200). **Basic candidate-vs-best gate only — the regression
  panel vs past champions is a deliberate follow-on, not yet wired.**
- **Tooling (`chessckers` `78cb26e`):** `gauntlet.py --no-curve/--out`; `cc strength` wired.
- Self-play pauses ~3–4 min per gate match (the cost of in-fleet gating).

## Log

- `06-28` Cold-launched (`cc fresh-run V6_e8d8_gated`). Trainer verified `lr=1e-3 / v5 c48 b5 /
  random init`; self-play **~14,000 games/day**. A startup `next_game` "Internal error 1" was an
  8 s race (client up before the first net) and self-resolved.
- `06-28` **Gate validated end-to-end.** net 1 bootstrap-promoted; net 2 queued a match; `match 1
  done: 21-18-1, calcElo=26 > −20 → PROMOTED` (40 `/match_result` games played). Every link works.
- `06-28` `cc strength` (Phase 1) validated locally too: run 7 vs run 6 = 50% / no regression
  (2-game smoke), early support for the v6 "barely loses strength" bet.

## Result

<infra goal met (gate + cc strength work). Leave the training "result" empty — this run isn't
testing a training change; it converges like run 7. Open follow-ons: regression-ladder panel in
the gate; a bigger run7-vs-run6 strength match; then a real unknown-answer experiment for run 9.>
