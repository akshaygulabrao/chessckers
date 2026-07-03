# Run 14 — official full starting position (normal games), warm-started from run 13

> The curriculum graduates back to the **complete game**: full FIDE White vs all 24 Black towers
> from the official start, with the opening double-move. Runs 11→13 built competence on hand-picked
> sub-positions (e8/d8 → d6/e6/f6 pawn-wall → d6/e6/f6 full army); run 14 is the same c64/b6 net let
> loose on the real Chessckers start. Same start FEN as run 10 (`STARTING_FEN`), but warm-started
> from run 13's stronger, larger net instead of run 9's.

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `V5_fullstart_c64b6` |
| Start FEN | **official full start** (= PyVariant `STARTING_FEN`): `pppppppp/pkkkkkkp/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR[a6:s,…,h6:s,a7:s,b7:k,…,g7:k,h7:s,a8:s,…,h8:s] w KQkq - 0 1 {wm:2}` — full FIDE White (R N B Q K B N R + 8 pawns) vs all 24 Black towers on ranks 6–8, **White to move with the opening double-move** (`{wm:2}`). Validated with `PyVariantClient().new_game()`: **matches `STARTING_FEN` exactly, 20 legal** White opening moves. Compiled into the fork (`src/chess/board.cc kStartposFen`) → **fork rebuild required**. Identical to run 10's start position. |
| Arch | SE-ResNet gather head, **c_filters=64, n_blocks=6, ~630K params**, tag `v5` (same as runs 11–13) |
| Optimizer | Adam, lr=1e-3 (flat; warmup/decay=0) — same as runs 11–13 |
| Rules | v6 bottom-*d* charge (same as runs 7–13) |
| **Init** | **WARM-START from run 13's net** (box seed `/workspace/run13_seed/weights.pt`, off-box backup `~/chessckers-backups/run13-army-d6e6f6-c64b6-20260702/`). Run 13's White learned to win a full army vs d6/e6/f6; run 14 tests whether that opening/coordination skill carries to the full 24-tower board. |
| **Replay buffer** | **fresh/empty** — wiped by `reset_fleet`; rebuilds from new run-14 self-play (window ramp 400→4000g @α0.75). Run 13's buffer is d6/e6/f6 (3-tower) self-play, off-distribution for the 24-tower start — only the *net* transfers (same reasoning as runs 10, 13). |
| Gate | fresh — DB bootstrapped `training_run #1 "V5_fullstart_c64b6"`; first published net bootstrap-promotes as best. 40-game candidate-vs-best on the new lineage. |
| Fleet box | vast id `42618148` (RTX 3060) — server `:39411`→`localhost:10100` (same box as runs 11–13) |
| Started | 2026-07-02 |
| Status | **active** — launching 2026-07-02 via `cc fresh-run` (warm from run 13, c64/b6, full official start). |

## Hypothesis

The sub-position campaign (runs 11–13) proved: c64/b6 solves e8/d8 (run 11), d6/e6/f6 is a
Black-forced win vs a lone king (run 12), and a **full White army beats d6/e6/f6** (run 13, White
91%). Run 14 asks the real question the whole campaign is building toward: **from the official full
starting position — all 24 towers, full chess — what does the net learn, and who is favored?**

- **Why warm-start from run 13 (not run 10 or cold)?** Run 10 already showed the full start is
  learnable when warm-started (its net beat the run-9 seed 10–0), but run 10 was c48/b5 off a d6/e6/f6
  endgame net. Run 13's net is c64/b6 and has seen a **full White army maneuver against Black towers**
  — much closer to the full-board opening than any prior seed. The board encoding is identical, so the
  weights load; the bet is that run 13's army-vs-towers features transfer better than run 9's did.
- **Why fresh buffer?** Run 13's buffer is 3-tower (d6/e6/f6) self-play — off-distribution for a
  24-tower opening. Keeping it would poison early training (the run-11 buffer-borrow mistake). Only the
  net transfers.
- **Is the full start a Black win?** Still open (run 10 couldn't answer it — a sweep over an OOD seed
  isn't an absolute strength signal). Self-play balance settling + sane opening lines in `cc games` is
  the first-order read; a real strength verdict needs a gauntlet vs earlier snapshots (deferred).

## Design delta vs run 13

- **New start position** — `board.cc kStartposFen` → the official full `STARTING_FEN` (full FIDE White
  + all 24 Black towers; `{wm:2}` opening double-move; `KQkq` castling). Needs a fork rebuild. Same FEN
  the fork already parsed in run 10, so no new parse path.
- **Warm-start** from run 13's net (`--base=/workspace/run13_seed/weights.pt`) instead of run 12's.
- Everything else identical: v5 **c64/b6 ~630K**, Adam 1e-3, v6 charge rule, in-fleet gate
  (calcElo>−20, 40-game), window ramp 400→4000g, `@reboot` auto-restart cron.

## Launch procedure (2026-07-02)

1. **Back up run 13's net** off-box (`~/chessckers-backups/run13-army-d6e6f6-c64b6-20260702/`) + to box
   seed `/workspace/run13_seed/weights.pt` (+ `.arch.json`) so it survives `reset_fleet` and serves as
   `--base`.
2. **Edit local `board.cc kStartposFen`** → the official `STARTING_FEN`; validate in PyVariant (matches
   `STARTING_FEN`, 20 legal White opening moves).
3. **`cc fresh-run`** warm-start (box already provisioned/built for c64/b6 from runs 11–13; the c64/b6
   arch dims are now plumbed through `cc fresh-run` per the run-13 patch): rsyncs the new board.cc,
   rebuilds the fork, `reset_fleet` (wipes run-13 DB/networks/games/buffer), relaunches server + trainer
   + client warm from the run13 seed.
4. Verify produced games start at the full official position; trainer log shows warm-load + `c=64 b=6`;
   `@reboot` cron `RUN_NAME` updated.

## Log

- `07-02` **Staged.** Closed run 13 as **White 91%** (a full army breaks the d6/e6/f6 Black-forced win).
  Backed up run 13's net to box `/workspace/run13_seed/` + Mac
  `~/chessckers-backups/run13-army-d6e6f6-c64b6-20260702/`. Edited `board.cc` → official `STARTING_FEN`;
  validated (matches `STARTING_FEN`, 20 legal). Launching via `cc fresh-run
  --run-name=V5_fullstart_c64b6 --arch=v5 --c-filters=64 --n-blocks=6 --se-ratio=8
  --base=/workspace/run13_seed/weights.pt`.

## Result

<active — leave empty. End-state TBD: from the official full start, does the run-13-seeded c64/b6 net
settle into a coherent, decisive balance (and which side), and how does it compare to run 10's c48/b5
full-start net? Link successor run when pivoted.>
