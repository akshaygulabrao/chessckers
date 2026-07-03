# Run 13 — full White FIDE army vs three KK towers (d6/e6/f6), warm-started from run 12

> The next curriculum rung after run 12 proved **d6/e6/f6 is a Black-forced win for a lone Ke1 + pawn
> wall**. Run 13 keeps Black identical (three 2-King towers on d6/e6/f6, the "6 kings") but gives
> **White its complete FIDE starting army** (back rank + pawns) and the opening double-move — asking
> whether real material lets White survive/win where the lone king could not.

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `V5_army_d6e6f6_c64b6` |
| Start FEN | **full White army vs d6/e6/f6 KK**: `8/8/3kkk2/8/8/8/PPPPPPPP/RNBQKBNR[d6:kk,e6:kk,f6:kk] w KQkq - 0 1 {wm:2}` — full FIDE White (R N B Q K B N R + 8 pawns) vs three 2-King towers on d6/e6/f6, **White to move with the opening double-move** (`{wm:2}`). Validated with `PyVariantClient().new_game()`: **16 legal** (4 knight + 12 pawn; the b/d/f/h pawns can't double — b4/d4/f4/h4 open a rim-pivot capture-chain onto Ke1, so they're Chessckers-check). Compiled into the fork (`src/chess/board.cc kStartposFen`) → **fork rebuild required**. Castling field `KQkq` matches run 10 / the official start (Black `kq` is inert — Black never generates castling). |
| Arch | SE-ResNet gather head, **c_filters=64, n_blocks=6, ~630K params**, tag `v5` (same as runs 11–12) |
| Optimizer | Adam, lr=1e-3 (flat; warmup/decay=0) — same as runs 11–12 |
| Rules | v6 bottom-*d* charge (same as runs 7–12) |
| **Init** | **WARM-START from run 12's net** (box seed `/workspace/run12_seed/weights.pt`, off-box backup `~/chessckers-backups/run12-d6e6f6-c64b6-<date>/`). Black side transfers ~exactly (identical tower setup + mating task); only White's material is new. |
| **Replay buffer** | **fresh/empty** — wiped by `reset_fleet`; rebuilds from new run-13 self-play (window ramp 400→4000g @α0.75). Only the *net* transfers. |
| Gate | fresh — DB bootstrapped `training_run #1 "V5_army_d6e6f6_c64b6"`; first published net bootstrap-promotes as best. 40-game candidate-vs-best on the new lineage. |
| Fleet box | vast id `42618148` (RTX 3060) — server `:39411`→`localhost:10100` (same box as runs 11–12) |
| Started | 2026-07-02 |
| Status | **done** — a full White army **breaks run 12's Black-forced win**: self-play settled at **White 91% / Black 9%** (~5.7k games, 56 nets). → run 14 (official full start). |

## Hypothesis

Run 12 proved d6/e6/f6 is a **Black-forced win** when White has only a lone king + pawn wall. Run 13
asks the natural follow-up: **does a full White army change the verdict?** With the back rank
(pieces to develop, a queen, castling) plus the opening double-move, White has genuine counterplay it
lacked in run 12.

- **Why warm-start (not cold)?** Black's setup and task are *identical* to run 12, so its policy/value
  transfer directly (run 12's Black is already ~96%-competent), and the trunk's board features carry
  over. Precedent: run 9→10 warm-started from a net that had concluded "Black wins d6/e6/f6" and White
  still learned to sweep the full official start 10–0 once it had material — so a Black-winning prior
  does **not** trap White when it's handed real pieces.
- **Why fresh buffer?** Run-12's replay buffer is lone-king/pawn-wall self-play — off-distribution for a
  full-army White. Keeping it would poison early training (the buffer-borrow mistake from run 11). Only
  the net transfers.
- Success = White's self-play share climbs off run 12's ~4% floor (and/or game length grows as White
  survives), i.e. real material is a genuine resource here. A continued ~96% Black would say the three
  KK towers beat even a full army from this setup.

## Design delta vs run 12

- **New start position** — `board.cc kStartposFen` → the full-army FEN above (full FIDE White back rank
  + pawns; Black unchanged at d6/e6/f6 KK; `{wm:2}` opening double-move; `KQkq` castling). Needs a fork
  rebuild. The fork already parsed a full army + castling + `{wm:2}` in run 10, so no new parse path.
- **Warm-start** from run 12's net (`--base=/workspace/run12_seed/weights.pt`) instead of run 12's
  run-11 base.
- Everything else identical: v5 **c64/b6 ~630K**, Adam 1e-3, v6 charge rule, in-fleet gate
  (calcElo>−20, 40-game), window ramp 400→4000g, `@reboot` auto-restart cron.

## Launch procedure (2026-07-02)

Surgical warm-start restart (box already provisioned/built for c64/b6 from runs 11–12 — NOT
`cc fresh-run`, which re-provisions + doesn't plumb c64/b6):

1. **Back up run 12's net** off-box + to box seed `/workspace/run12_seed/weights.pt` (+ `.arch.json`) so
   it survives `reset_fleet` and can serve as `--base`.
2. **Stop self-play** (`tmux kill-session -t cc-client` + `pkill akshay-chessckers-0`) to free the
   engine binary (else the relink hits ETXTBSY).
3. **Edit `board.cc kStartposFen`** → the run-13 FEN (local fork + rsync to box), **rebuild**
   `ninja -C build/release akshay-chessckers-0`.
4. **Stop server+trainer** (`tmux kill-session -t cc`), then **`reset_fleet.sh`** — wipe
   DB/networks/games/pgns/`trainer/run*` + client net-cache (the run-12 buffer).
5. **Relaunch** warm: `RUN_NAME=V5_army_d6e6f6_c64b6 ARCH_VERSION=v5 C_FILTERS=64 N_BLOCKS=6 SE_RATIO=8
   BASE=/workspace/run12_seed/weights.pt restart_fleet.sh`.
6. Update the `@reboot` cron `RUN_NAME` label; verify produced games start at the full-army position.

## Log

- `07-02` **Staged.** Validated the start FEN in PyVariant (16 legal: 4 knight + 12 pawn; the
  dark-square b/d/f/h pawn-doubles are Chessckers-check via a rim-pivot capture-chain from the d6 tower
  onto Ke1). Closed run 12 as a Black-forced win. Ledger written; box cutover pending.
- `07-02` **Launched + verified** via `cc fresh-run --run-name=V5_army_d6e6f6_c64b6 --arch=v5
  --c-filters=64 --n-blocks=6 --se-ratio=8 --base=/workspace/run12_seed/weights.pt`. Backed up run 12's
  net to box `/workspace/run12_seed/` + Mac `~/chessckers-backups/run12-d6e6f6-c64b6-20260702/`.
  **Patched `cc fresh-run`** to plumb `--c-filters/--n-blocks/--se-ratio` (it previously only passed
  `ARCH_VERSION`, so a c64/b6 warm-start would have crashed on shape mismatch). Trainer log confirms
  `arch=v5 c=64 b=6`, `warm-started from /workspace/run12_seed/weights.pt` (clean load, no mismatch);
  server `Bootstrapped training run #1 ("V5_army_d6e6f6_c64b6")`, net id=1 bootstrap-promoted. Fresh
  buffer + fresh optimizer (reset_fleet wiped the run-12 snapshot). `@reboot` cron updated with the
  c64/b6 env.

## Result

**Yes — a full White army breaks run 12's Black-forced win from d6/e6/f6.** Self-play balance
inverted from run 12's ~96% Black to **White 91% / Black 9% / draw 0%** (~5.7k games, 56 published
nets, best #56 `f31d595c…`). With the back rank, queen, castling and the opening double-move, White's
material is a genuine, decisive resource where the lone Ke1 + pawn wall (run 12) could not survive.
The warm-start from run 12's net transferred cleanly and did **not** trap White in the Black-winning
prior — same pattern as run 9→10.

- **Answered:** the three KK towers on d6/e6/f6 do **not** beat a full FIDE army. The curriculum rung
  is complete — White has enough to win this sub-position.
- **→ run 14:** graduate back to the **official full starting position** (all 24 Black towers + full
  White army), warm-started from run 13's net, same c64/b6 / Adam 1e-3. Run 13's net (box seed
  `/workspace/run13_seed/weights.pt`, Mac `~/chessckers-backups/run13-army-d6e6f6-c64b6-20260702/`) is
  the run-14 base.
