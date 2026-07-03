# Run 12 — d6/e6/f6 pawn-wall, warm-started from run 11

> First step up from the **solved** e8/d8 KK-vs-K endgame (run 11) toward the full game.
> Takes run 11's competent **c64/b6** net and warm-starts it onto the **d6/e6/f6 pawn-wall**
> start (the run-9 position) — now at the bigger scale. The endgame→opening-scaffold rung
> in the curriculum (e8/d8 → d6/e6/f6 → … → full start).

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `V5_d6e6f6_c64b6` |
| Start FEN | **d6/e6/f6 pawn-wall**: `8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] w - - 0 1` — White's full pawn wall (rank 2) + Ke1 vs three 2-King towers on d6/e6/f6, **White to move** (16 legal; validated with `PyVariantClient().new_game()`). Compiled into the fork (`src/chess/board.cc kStartposFen`) → **fork rebuilt**. Same position as run 9. |
| Arch | SE-ResNet gather head, **c_filters=64, n_blocks=6, ~630K params**, tag `v5` (same as run 11) |
| Optimizer | Adam, lr=1e-3 (flat; warmup/decay=0) |
| Rules | v6 bottom-*d* charge (same as runs 7–11) |
| **Init** | **WARM-START from run 11's net** (box seed `/workspace/run11_seed/weights.pt`, off-box backup `~/chessckers-backups/run11-e8d8-c64b6-<date>/`), not cold. |
| **Replay buffer** | **fresh/empty** — wiped by `reset_fleet`; rebuilds from new d6/e6/f6 self-play (window ramp 400→4000g @α0.75). Only the *net* transfers, not the e8/d8 buffer. |
| Gate | fresh — DB bootstrapped `training_run #1 "V5_d6e6f6_c64b6"`; first published net bootstrap-promotes as best. 40-game candidate-vs-best on the new lineage. |
| Fleet box | vast id `42618148` (RTX 3060) — `ssh -p 39056 root@171.248.168.109`, server `:39411`→`localhost:10100` (same box as run 11) |
| Started | 2026-07-01 |
| Status | **done** — Black wins by force from d6/e6/f6 even at c64/b6 (~96% Black, held flat; White could not hold as the human either). Superseded → run 13. |

## Hypothesis

Run 11 gave a competent c64/b6 net on the solved **e8/d8 KK-vs-K** endgame — the first step toward
competent play. **Run 12 tests the next rung:** does that endgame skill warm-start cleanly onto the
**d6/e6/f6 pawn-wall** (run 9's position, but at the bigger scale)? Run 9 already established that
d6/e6/f6 is a **Black win** (~99% Black) with the small c48/b5 net; run 12 asks whether the scaled net,
seeded from a related endgame, reaches the same conversion — ideally faster than run 9's cold-ish path.

- **Why warm-start (not cold)?** The encoding is identical (e8/d8 and d6/e6/f6 share the input format),
  so run 11's weights load directly. Warm-starting from a net that already coordinates KK-towers-vs-king
  should transfer the mate skill and skip near-random early play — the same lever that worked run 9→10.
- **Why fresh buffer?** The e8/d8 replay buffer is off-distribution for a pawn-wall start; keeping it
  would poison early training (the buffer-borrow mistake that killed run 11's first attempt). Only the
  net transfers.
- Success = the c64/b6 net converges on d6/e6/f6 (≳ run 9's ~99% Black), confirming the endgame→scaffold
  warm-start rung of the curriculum holds at the bigger scale.

## Design delta vs run 11

- **New start position** — `board.cc kStartposFen` → the d6/e6/f6 pawn-wall FEN (validated in PyVariant:
  White to move, 16 legal, clean roundtrip). Needs a fork rebuild.
- **Warm-start** from run 11's net (`--base=/workspace/run11_seed/weights.pt`) instead of run 11's cold
  random init.
- Everything else identical: v5 **c64/b6 ~630K**, Adam 1e-3, v6 charge rule, in-fleet gate (calcElo>−20,
  40-game), window ramp 400→4000g, `@reboot` auto-restart cron.

## Launch procedure (2026-07-01)

Surgical warm-start restart (box already provisioned/built for c64/b6 from run 11 — NOT `cc fresh-run`,
which re-provisions + doesn't plumb c64/b6):

1. **Back up run 11's net** off-box + to box seed `/workspace/run11_seed/weights.pt` (+ `.arch.json`) so
   it survives `reset_fleet` and can serve as `--base`.
2. **Stopped self-play** (`tmux kill-session -t cc-client` + `pkill akshay-chessckers-0`) to free the
   engine binary (else the relink hits ETXTBSY).
3. **Edited `board.cc kStartposFen`** → the d6/e6/f6 FEN (local fork + rsync to box), **rebuilt**
   `ninja -C build/release akshay-chessckers-0`.
4. **Stopped server+trainer** (`tmux kill-session -t cc`), then **`reset_fleet.sh`** — wiped
   DB/networks/games/pgns/`trainer/run*` + client net-cache (the e8/d8 buffer).
5. **Relaunched** warm: `RUN_NAME=V5_d6e6f6_c64b6 ARCH_VERSION=v5 C_FILTERS=64 N_BLOCKS=6 SE_RATIO=8
   BASE=/workspace/run11_seed/weights.pt restart_fleet.sh` → trainer warm-started from run 11's net,
   fresh server run #1, self-play client pulled the new net (`--visits=800 --backend=chessckers CUDA`).
6. Updated the `@reboot` cron `RUN_NAME` label; verified produced games start at d6/e6/f6.

## Log

- `07-01` **Launched + verified.** Backed up run 11's net (630,325 params, first conv `(64,16,3,3)`
  → confirmed c64/b6) to box `/workspace/run11_seed/weights.pt` (+ arch sidecar) and off-box
  `~/chessckers-backups/run11-e8d8-c64b6-20260701/`. Stopped self-play, rsync+rebuilt the fork with the
  d6/e6/f6 FEN (`ninja` relinked `board.cc` only), stopped server+trainer, `reset_fleet` (wiped the
  e8/d8 DB/nets/games/buffer), relaunched warm. Trainer log confirms `warm-started from
  /workspace/run11_seed/weights.pt`, `arch=v5 c=64 b=6`, `device=cuda`, fresh `window=400→4000g` ramp
  (fresh buffer). Server bootstrapped `run #1 "V5_d6e6f6_c64b6"`; net id=1 bootstrap-promoted as best.
  First self-play game renders the pawn-wall opening (White `e2e3 d2d4 c2c4`, Black `f6g7 d6e6` from
  d6/e6/f6, result `0-1` Black) — confirms the rebuilt fork plays from the new start. `@reboot` cron
  RUN_NAME updated to `V5_d6e6f6_c64b6` (+ c64/b6 env).

## Result

**Black wins by force.** Warm-started from run 11 at c64/b6, the balance settled at **~White 4% /
Black 96% / draw 0%** and held flat — no upward White trend across training, and game length stayed
flat (~24 plies median, no survival growth). The human operator also could not hold the White side.
So d6/e6/f6 (lone Ke1 + full pawn wall vs three KK towers) is a **Black-forced win** at this scale,
confirming run 9's small-net read (~99% Black) carries to the bigger c64/b6 net. The run-11 warm-start
transferred cleanly (coherent pawn-wall opening from game 1).

Pivoted to **run 13**: same three KK towers on d6/e6/f6, but give **White its full FIDE army**
(`RNBQKBNR` + pawns) plus the opening double-move — testing whether real material lets White survive/win
where the lone king could not.
