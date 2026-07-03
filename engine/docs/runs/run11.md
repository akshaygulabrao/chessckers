# Run 11 — c64/b6 scale-up on the e8/d8 KK-vs-K endgame (cold, fresh buffer)

> **Supersedes the abandoned buffer-borrow experiment.** Run 11 was originally a c48/b5→c64/b6
> capacity scale-up on the *full official start* that **kept run 10's replay buffer** while
> cold-starting the weights. That was judged **a mistake** (borrowing the buffer conflated a
> scale-up with a distillation and pinned the new net's ceiling to run-10 strength). It was
> **terminated 2026-06-30** and run 11 was re-cast as a **clean cold restart** on the
> well-understood **e8/d8 KK-vs-K endgame** at the bigger c64/b6 scale — no borrowed data.
> (For the discarded attempt, see the `git` history of this file / the `chessckers-buffer-preserving-scaleup` memory.)

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `V5_e8d8_c64b6` |
| Start FEN | **e8/d8 KK-vs-K**: `3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] b - - 0 1` — two 2-King towers on d8 & e8 vs White's lone king on e1, **Black to move** (22 legal tower moves; validated with `PyVariantClient().new_game()`). Compiled into the fork (`board.cc kStartposFen`) → **fork rebuilt**. |
| Arch | SE-ResNet gather head, **c_filters=64, n_blocks=6, ~630K params**, tag `v5` |
| Optimizer | Adam, lr=1e-3 (flat; warmup/decay=0) |
| Rules | v6 bottom-*d* charge (same as runs 7–10) |
| **Init** | **cold fresh RANDOM init** (`BASE=""`) — after `reset_fleet` wiped `weights.pt`, `launch_trainer` auto-BASE finds nothing → random. |
| **Replay buffer** | **fresh/empty** — the borrowed run-10 buffer was **wiped** by `reset_fleet`. Rebuilds from new e8/d8 self-play (window ramp 400→4000g @α0.75). |
| Gate | fresh — DB bootstrapped `training_run #1 "V5_e8d8_c64b6"`; first published net bootstrap-promotes as best. 40-game candidate-vs-best on the new lineage. |
| Fleet box | vast id `42618148` (RTX 3060) — `ssh -p 39056 root@171.248.168.109`, server `:39411`→`localhost:10100` |
| Started | 2026-06-30 |
| Status | **done** — converged on e8/d8 (~99% Black over ~81.8k games); first step toward competent play. Superseded → run 12. |

## Hypothesis

Runs 5–8 converged the **e8/d8 KK-vs-K** conversion at **c48/b5 v5 (~364K)** — the tiny c16/b1 net
even learned it ("extremely good training"). Run 11 tests whether the **bigger c64/b6 body
(~630K, ~1.73× params)** learns the same well-understood endgame at least as cleanly (and how the
extra capacity affects games-to-converge / final policy quality) as a **clean baseline** for future
c64/b6 work — this time **without** the run-11-original buffer-borrow confound.

- **Why e8/d8?** It's a known, satisfying convergence target (runs 5–8) that **search cannot
  shortcut** (move count explodes past the visit budget → the net must *learn* the conversion), so it
  isolates whether the scaled net trains well from scratch. See [[chessckers-e8d8-endgame-run]].
- **Why cold + fresh buffer?** The whole point of terminating the original run 11 was that a borrowed
  buffer was the wrong experiment. A fresh random init on fresh e8/d8 self-play is the honest test of
  the c64/b6 net.
- Success = the c64/b6 net converges on e8/d8 (≳ the c48/b5 runs' ~99% Black), giving a trustworthy
  bigger-net baseline.

## Cold-restart procedure (2026-06-30)

Surgical cold restart — the box was already provisioned/built for c64/b6 (from the original run 11),
so only the **compiled start FEN** changed + a full wipe. NOT `cc fresh-run` (would re-provision +
rebuild server/client unnecessarily and doesn't plumb c64/b6):

1. **Stopped self-play** (`tmux kill-session -t cc-client` + `pkill akshay-chessckers-0`) to free the
   engine binary (else the relink hits ETXTBSY).
2. **Edited `board.cc kStartposFen`** → the e8/d8 FEN (local fork + rsync to
   `/workspace/chessckers/akshay-chessckers-0`), **rebuilt** `ninja -C build/release akshay-chessckers-0`
   (binary mtime bumped; client symlinks `.enginebin/akshay-chessckers-0` → that build).
3. **Stopped server+trainer** (`tmux kill-session -t cc`), then **`reset_fleet.sh`** — wiped
   DB/networks/games/pgns/`trainer/run*` + client net-cache (**the borrowed buffer went here**).
4. **Relaunched** `RUN_NAME=V5_e8d8_c64b6 ARCH_VERSION=v5 C_FILTERS=64 N_BLOCKS=6 SE_RATIO=8 restart_fleet.sh`
   → cold trainer (`base=<random init>`, `c=64 b=6`), fresh server run #1, self-play client pulled the
   new net (`--visits=800 --backend=chessckers CUDA`).
5. Updated the `@reboot` cron `RUN_NAME` label; verified produced games start at e8/d8.

Box layout note: single fork at `/workspace/chessckers/akshay-chessckers-0` (build dir `build/release`),
NOT the `/workspace/akshay-chessckers-0` the old cold-restart memory cites — layout drifted.

## Log

- `07-01` **Declared done.** Fleet fully converged the e8/d8 conversion at c64/b6: `cc status`
  showed **~81.8k games, balance White 1% / Black 99% / draw 0%**, 594 nets published (585 promoted
  via the in-fleet gate), trainer at step 8446. The bigger c64/b6 body learned the well-understood
  endgame as cleanly as the c48/b5 runs (5–8) — a trustworthy scaled baseline. This was **the first
  step toward competent play**; pivoting to run 12 (run 11's net warm-started onto the d6/e6/f6
  pawn-wall start).
- `06-30` **Terminated the buffer-borrow run 11** (user: "borrowing the replay buffer was a huge
  mistake") and **cold-restarted run 11 on e8/d8 at c64/b6**. Validated the e8/d8 FEN in PyVariant,
  edited+rebuilt the fork, `reset_fleet` (wiped the borrowed buffer + run-11 gate history), relaunched
  cold via `restart_fleet.sh`. Trainer up: `base=<random init>`, `arch=v5 c=64 b=6`, published initial
  EMA weights fresh; server bootstrapped `run #1 "V5_e8d8_c64b6"`; client running self-play on the new
  net. Run 10's best net remains backed up (Mac `~/chessckers-backups/run10-fullstart-20260630/` +
  box `/workspace/run10_seed/`).

## Result

**Converged.** At ~81.8k self-play games the balance sat at **White 1% / Black 99% / draw 0%** — i.e.
Black reliably converts the two KK-towers vs White's lone king from the e8/d8 start, matching the
c48/b5 runs (5–8, ~99% Black). 594 nets published, 585 promoted through the in-fleet gate. The scaled
**c64/b6 (~630K) body learns the endgame at least as cleanly as c48/b5**, so it stands as the
trustworthy bigger-net baseline this run set out to establish. **First step toward competent play** —
pivoted to **run 12**: run 11's net warm-started onto the d6/e6/f6 pawn-wall start.
