# Chessckers training & analysis toolkit

Everything you need to **launch a run on vast, watch it train, diagnose whether
it's learning or stuck, and design/validate the next start position.** Built
from the workflow of the e8/d8 endgame run (which converged — Black solves the
mate at ~99.5%).

## The `cc` command-center

`cc` auto-resolves the live vast.ai box (via `vastai show instances`) so **no
command hardcodes an ssh endpoint** — important because the port/IP change every
time the instance is recreated. Set up a shortcut once:

```bash
alias cc='/Users/ox/AAworkspace/chessckers/engine/.venv/bin/python /Users/ox/AAworkspace/chessckers/engine/scripts/cc.py'
```

Then, from **any directory**:

| Command | What it does | Runs on |
|---|---|---|
| `cc box` | show the resolved box (ssh, server URL, paths) | local |
| `cc ssh [cmd]` | shell on the box, or run one command | box |
| `cc run <script.py> [args]` | run any `engine/scripts/` script on the box | box |
| `cc doctor` | **one-shot health + convergence report** | box |
| `cc plot` | terminal sparklines of the run's curves | box |
| `cc validate "<FEN>"` | is this start winnable for Black? mate length? | local |
| `cc launch` | print the fresh-run runbook | local |

> Multiple boxes running? `CC_INSTANCE=<id> cc ...` picks one. `cc box --refresh`
> re-queries vast (cached 10 min).

## 1. Launching a run

The fleet is the lc0 ecosystem on the box: `cc-server` (collects games, serves
nets) + `trainer_bridge → chessckers_engine.train_continuous` (this repo) +
`akshay-chessckers-0` self-play clients. To start a fresh run **on the box**:

```bash
# set the start position (one mechanical step the helper does safely):
FEN='3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] b - - 0 1' CONFIRM=yes \
  lczero-server/scripts/new_run.sh          # rewrites kStartposFen in board.cc

# rebuild the fork engine (board.cc changed) — see chessckers-engine-cuda-linux-port memory
# then (DESTRUCTIVE — wipes the prior run):
lczero-server/scripts/reset_fleet.sh
lczero-server/scripts/run_server_vast.sh          # server + trainer bridge
lczero-client/scripts/launch_vast_direct.sh       # self-play
```

Trainer knobs are env vars on `launch_trainer.sh` (`LR`, `WINDOW_GAMES`,
`PUBLISH_GAMES`, `EMA_DECAY`, `MIN_BUFFER`); `--grad-clip`/`--momentum` live in
`train_continuous.py` (not plumbed through the bridge).

**Before** committing a run to a new start position, check it's actually a forced
Black win and how deep:

```bash
cc validate "<your start FEN>"     # -> "WINNABLE ✓ box-shrink Black mates in N plies" or not
```

## 2. Monitoring

```bash
cc doctor                          # procs · trainer step/rate/lr · agreement · W/B/draw trend · VERDICT
cc plot                            # convergence sparkline + training-signal sparklines
```

`cc doctor` verdicts: `TOO EARLY` · `TRAINING` · `LEARNING / IN TRANSITION`
(win-rate moving) · `STALLED / SATURATED` (one side ≥90% flat for many blocks —
the cold-start trap) · `CONVERGED`.

**Time-series (the "no metrics store" fix):** run the doctor as a sampler in a
loop to accumulate a CSV, then plot it:

```bash
# on the box, append a row every 5 min:
while true; do cc run run_doctor.py --csv run_metrics.csv >/dev/null; sleep 300; done
```

The fleet's own dashboard is `lczero-server/scripts/fleet_status.py`
(`cc run ../lczero-server/scripts/fleet_status.py` — note: lives outside engine).
Its "latest Xm ago" field was lexicographically buggy and is now fixed.

## 3. Diagnosing "is it learning, or stuck?"

The decisive tool — replays **every checkpoint** against a frozen probe suite to
reconstruct the strength-vs-time curve (no waiting):

```bash
cc run gen_probe_suite.py ../lczero-server/pgns/run1 probe_suite.jsonl 40 211   # freeze the yardstick
cc run eval_history.py    ../lczero-server/trainer/run1 probe_suite.jsonl --sims 2 --device cuda
```

Rising mass at the latest checkpoints ⇒ **undertrained** (keep training); a long
flat plateau ⇒ **stuck** (needs intervention: a retrograde curriculum, LR decay,
or more exploration). Supporting views:

```bash
cc run game_phase_stats.py     ../lczero-server/pgns/run1 400    # early/mid/late: result, length, material, captures
cc run backrank_check_trend.py ../lczero-server/pgns/run1 300 10 # is Black learning the camp-denying check?
```

## 4. Solving / designing positions

`solve_endgame.py` (local, pure rules):

```bash
cc validate "<FEN>"                                  # winnable? mate length?
.venv/bin/python scripts/solve_endgame.py --human    # YOU play Black vs heuristic White
.venv/bin/python scripts/solve_endgame.py --play     # watch the box-shrink heuristic auto-mate
.venv/bin/python scripts/solve_endgame.py --max-depth 8   # forced-mate prover (feasible only ~8-10 ply)
```

The **box-shrink** Black policy (drive the king to the rim/corner + coordinate
both towers, don't stalemate/hang) is the reference mating procedure; `--w-*`
flags tune its weights. It auto-mates the e8/d8 start in ~6 moves.

## Script index

| Script | Where | Purpose |
|---|---|---|
| `cc.py` | local | command-center / vast resolver |
| `run_doctor.py` | box | consolidated status + convergence verdict (+ `--csv` sampler) |
| `plot_run.py` | box | terminal sparklines of the run curves |
| `eval_history.py` | box | strength-vs-time over all checkpoints (undertrained-vs-stuck) |
| `gen_probe_suite.py` | box | freeze the eval probe suite from real games |
| `game_phase_stats.py` | box | early/mid/late game characterization |
| `backrank_check_trend.py` | box | back-rank-check learning trend |
| `solve_endgame.py` | local | solver / box-shrink policy / `--human` / `--validate` |
| `lczero-server/scripts/new_run.sh` | box | guarded fresh-run setup (sets start FEN) |

## Gotchas (hard-won)

- **Endpoint changes on instance recreate** — always go through `cc` (or
  `vastai show instances`), never a hardcoded `ssh5.vast.ai:NNNNN`.
- **Self-play can't discover a deep, high-branching mate from scratch** — the
  e8/d8 run sat at ~1% Black for ~38k games (cold-start trap) before flipping to
  99%. It was *undertrained*, not stuck. Watch the `cc doctor` verdict and the
  `eval_history` curve to tell the difference; a retrograde curriculum (seed
  mate-in-k positions) is the fix if it's genuinely stuck.
- **LR is constant 0.02** (no warmup/decay) — gains don't anneal/consolidate;
  set `--lr-decay-steps`/`--lr-gamma` if you want convergence to lock in.
- **`reset_fleet.sh` is destructive**; restarting `cc-server` alone preserves
  state (DB + networks on disk).
