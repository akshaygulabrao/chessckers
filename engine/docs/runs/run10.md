# Run 10 — official full starting position, warm-started from run 9

> The **first run on the complete game**: full FIDE chess vs all 24 Black towers, with the
> opening double-move. Bootstrapped from run 9's net (which solved d6/e6/f6). The campaign's
> graduation from hand-built endgame sub-positions to the real Chessckers start.

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `V5_fullstart` |
| Start FEN | `pppppppp/pkkkkkkp/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR[a6:s,…,h6:s,a7:s,b7:k,…,g7:k,h7:s,a8:s,…,h8:s] w KQkq - 0 1 {wm:2}` — official start (= PyVariant `STARTING_FEN`), full chess vs 24 towers, **opening double-move** (`{wm:2}`). Compiled into fork `src/chess/board.cc` `kStartposFen`. |
| Arch | SE-ResNet gather head, c_filters=48, n_blocks=5, ~364K params, tag `v5` (same as run 9) |
| Optimizer | Adam, lr=1e-3 (same as run 9) |
| Rules | v6 bottom-*d* charge (same) |
| **Init** | **WARM-START from run 9's net** (`~/chessckers-backups/run9-d6e6f6-v6gated-20260629/weights.pt`; box seed `/workspace/run9_seed/weights.pt`), not cold |
| Gate | in-fleet lc0 gate live (calcElo > −20), 40-game candidate-vs-best |
| Fleet box | vast (resolve with `cc box`) — launched 2026-06-29 |
| Status | **active** — launched 2026-06-29 from the official start, warm-started from run 9 |

## Hypothesis

The endgame campaign (e8/d8 → d6/e6/f6, runs 6–9) showed Black can force wins from hand-built
sub-positions and that warm-starting transfers learned tower-coordination/mate skills. **Run 10
tests the real game:** can the fleet learn anything coherent from the *full* starting position,
warm-started from the d6/e6/f6 net?

Open questions:
1. **Is the full start even a Black win?** Unknown — Black has overwhelming material (24 towers)
   but White moves first with the double-move and a full board to maneuver. Self-play balance is
   the first-order signal (as in run 9, gate Elo will likely be flat for two copies of the same net).
2. **Does the d6/e6/f6 skill transfer to the full board?** The input encoding is identical, so the
   weights load; whether the learned features help on a 24-tower opening vs a 3-tower endgame is
   the experiment. A cold start here would plausibly never escape near-random — warm-start is the lever.

Success = a coherent, improving policy (decisive self-play balance settling, sane opening lines in
`cc games`), not necessarily a "solved" position on the first run.

## Design delta vs run 9

- **New start position** — `board.cc kStartposFen` → the official full `STARTING_FEN` (verified it
  parses + roundtrips through PyVariant; 20 legal White opening moves, double-move handled). Needs
  a fork rebuild (done by `cc fresh-run`, which rsyncs the local fork).
- **Warm-start** from run 9's net instead of run 8's (`--base=/workspace/run9_seed/weights.pt`).
- Everything else identical: v5 c48/b5, Adam 1e-3, v6 charge rule, in-fleet gate, `cc strength`,
  `@reboot` auto-restart cron.

## Log

- `06-29` Set up: run 9 declared done (Black solved d6/e6/f6 ~99%); its net backed up off-box +
  to `/workspace/run9_seed/weights.pt`; `board.cc` → official `STARTING_FEN`.
- `06-29` **Launched + verified.** `cc fresh-run --run-name=V5_fullstart --arch=v5 --parallelism=32
  --base=/workspace/run9_seed/weights.pt` (rebuilt the fork, reset_fleet, relaunched). Trainer log
  confirms `[train] warm-started from /workspace/run9_seed/weights.pt` (NOT random init; first net
  `f48657b3…` bootstrap-promoted, ≠ cold-init `cf42568…`). Recorded self-play game 1 renders the
  full board — full chess vs all 24 towers — so self-play runs from the official start. Server +
  bridge + trainer UP, games flowing, DB reset clean. Early balance White 0% / Black 100% (noise at
  4 games — Black's 24-tower material edge). `@reboot` auto-restart cron reinstalled.

## Result

<active — fill once it runs. First-order signal: self-play W/B balance + whether opening lines in
`cc games` look coherent. Open question: is the full start a Black win, and does the d6/e6/f6 net
transfer to the 24-tower opening?>
