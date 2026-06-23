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
| `cc status` | **fleet dashboard** — live arena + gate promote/reject decisions | box |
| `cc plot` | terminal sparklines of the run's curves | box |
| `cc ladder [opts]` | **round-robin champion nets → terminal Elo + score matrix** | box |
| `cc gauntlet [opts]` | **current net vs ALL previous snapshots** → strength + regression curve | box |
| `cc games [opts]` | **pull a RECORDED self-play game off the box + render its real moves** | box→local |
| `cc watch [opts]` | pull the latest fleet net + watch it self-play live | box→local |
| `cc restart-trainer [LR]` | **hardened clean warm-restart** (snapshot-guarded; optionally change LR) | box |
| `cc play [opts]` | **play a human-vs-net game** against the latest fleet net | box→local |
| `cc fresh-run [opts]` | **provision + launch a complete training run from scratch** | box |
| `cc launch` | print the fresh-run runbook (manual steps) | local |

`cc games` is the quick way to eyeball what the network is actually playing — it
fetches the chunk and renders the *actual sampled moves* (not the visit-argmax)
via `watch_game.py --chunk`:

```
cc games                  # newest recorded game, board move-by-move (no net needed)
cc games --list [K]       # list the K newest chunks with ages (default 15)
cc games --index N        # a specific training.N.gz
cc games --eval           # also pull the fleet net + show per-ply WDL
cc games --step           # any extra args (--step/--delay/--clear/--max-plies) pass through
cc watch --device mps     # latest net plays a fresh game from the start FEN
```

`cc restart-trainer` is the ONLY supported way to change a live hyperparameter (the
trainer reads `--lr` etc. once at startup). It captures the live deployment env from
the running bridge, Ctrl-Cs the trainer for a clean shutdown, waits for a **fresh**
`replay_buffer.pkl` (aborting rather than losing the ~4000-game window), relaunches
warm with the new LR, and verifies. The heavy lifting is `../lczero-server/scripts/restart_trainer.sh`
(run on the box) so no long command is ever pasted — a terminal soft-wrap newline in a
`tmux send-keys` line silently breaks the relaunch.

```
cc restart-trainer 0.002            # change LR to 0.002 and warm-restart
cc restart-trainer                  # warm-restart at the current LR (just kick it)
cc restart-trainer 0.002 --dry-run  # show the plan + captured env, touch nothing
```

> Multiple boxes running? `CC_INSTANCE=<id> cc ...` picks one. `cc box --refresh`
> re-queries vast (cached 10 min).

## 1. Launching a run

**One command (recommended):** `cc fresh-run` provisions the box, builds the fork
+ client, resets fleet state, and launches server/trainer/client — all 6 steps
in one shot.

```bash
cc fresh-run                          # defaults: V5_e8d8, v5 arch, 32 parallelism
cc fresh-run --run-name=V5_exp2 --arch=v5 --parallelism=64
```

**Manual steps** (what `cc fresh-run` automates):

```bash
# set the start position (one mechanical step the helper does safely):
FEN='3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] b - - 0 1' CONFIRM=yes \
  ../lczero-server/scripts/new_run.sh          # rewrites kStartposFen in board.cc

# rebuild the fork engine (board.cc changed) — see chessckers-engine-cuda-linux-port memory
# then (DESTRUCTIVE — wipes the prior run):
../lczero-server/scripts/reset_fleet.sh
../lczero-server/scripts/run_server_vast.sh          # server + trainer bridge
../lczero-client/scripts/launch_vast_direct.sh       # self-play
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

The fleet's own dashboard is `cc status` (it runs `../lczero-server/scripts/fleet_status.py`
on the box; that script lives outside engine, so `cc run` can't reach it). It shows the
LIVE arena match + the gate's recent promote/reject decisions — it used to hardcode
"arenas removed" (false since the 2026-06-13 re-enable). Its "latest Xm ago" field's
lexicographic bug is also fixed.

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

**Head-to-head strength (no oracle needed):** `cc ladder` plays champion snapshots
against each other and prints a pairwise score matrix + a Bradley-Terry Elo ranking
+ a chronological Elo curve — the oracle-free answer to "is the best actually beating
its past selves?", and it surfaces the non-transitivity (A>B>C>A) the probe-suite
curve can't. Runs on the box (nets + GPU are there).

```bash
cc ladder                              # ~6 snapshots sampled from the run dir, round-robin
cc ladder --n 8 --games 6 --sims 200   # more nets / games / search = less noise
cc ladder --vs-best                    # everyone vs the newest only (quick anchor ladder)
cc ladder a.pt b.pt c.pt               # explicit nets
```

**Current vs all its past selves** — `cc gauntlet` is the focused "is the live net
stronger than every net before it?" audit the lenient gate can't do: it plays the
current `weights.pt` vs each sampled `iter-async-*.pt` snapshot, prints a per-opponent
score + Elo + a strength curve (oldest→newest), and flags any older net that beats
current (a promoted regression). The published `.bin` champions aren't loadable in the
Python MCTS path, so the trainer checkpoint lineage stands in for "previous nets".

```
cc gauntlet                            # current vs ~6 sampled snapshots (~10-20 min)
cc gauntlet --n 16 --games 6 --sims 200
cc gauntlet --all                      # vs EVERY snapshot (hours)
```

(Slow: pure-Python PyVariant MCTS is CPU-bound and shares the box with the live
fleet — background a big run, or keep `--sims`/`--games`/`--n` modest.)

## 4. Playing the net

Play a human-vs-net game from any FEN — you pick from a numbered legal-move menu
(no hand-typing cadence/deploy UCI), the net replies via MCTS, and the board +
WDL eval render each ply:

```bash
cc play --color black                # play the LIVE fleet champion (pulls its net)
.venv/bin/python scripts/play_net.py --color white --sims 200 --device mps
.venv/bin/python scripts/play_net.py "<FEN>" --weights X.pt --color black
```

`--color` is the side YOU play (default black = the towers); at your turn `u`
undoes your last move and `q` quits. To watch the net play ITSELF instead, use
`watch_game.py` / `cc watch`.

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
| `ladder.py` | box | round-robin / vs-best Elo + score matrix over sampled nets |
| `gauntlet.py` | box | **current net vs all previous snapshots** — strength + regression curve |
| `play_net.py` | local | **human-vs-net** play from any FEN (numbered move menu) |
| `../lczero-server/scripts/new_run.sh` | box | guarded fresh-run setup (sets start FEN) |

## Gotchas (hard-won)

- **Endpoint changes on instance recreate** — always go through `cc` (or
  `vastai show instances`), never a hardcoded `ssh5.vast.ai:NNNNN`.
- **vast proxy SSH (`sshN.vast.ai:NNNNN`) is unreliable/dead** — use the DIRECT
  endpoint: `vastai ssh-url <id>` (public ip + the container-port-22 host
  mapping). `cc` resolves this automatically now; for the provision/launch
  scripts set `VAST_HOST`/`VAST_PORT` from `ssh-url`, never the proxy.
- **Trust the API's effective CPU count, not the box's `nproc`** — vast shows the
  *host's* full core count via `nproc` (e.g. 64) but only allocates a fraction
  (`cpu_cores_effective`, e.g. 16). Self-play is CPU-MCTS-bound, so choose the
  self-play box by the **API** count (`vastai show instances`), not `nproc`.
- **Self-play can't discover a deep, high-branching mate from scratch** — the
  e8/d8 run sat at ~1% Black for ~38k games (cold-start trap) before flipping to
  99%. It was *undertrained*, not stuck. Watch the `cc doctor` verdict and the
  `eval_history` curve to tell the difference; a retrograde curriculum (seed
  mate-in-k positions) is the fix if it's genuinely stuck.
- **LR is constant 0.02** (no warmup/decay) — gains don't anneal/consolidate;
  set `--lr-decay-steps`/`--lr-gamma` if you want convergence to lock in.
- **`reset_fleet.sh` is destructive**; restarting `cc-server` alone preserves
  state (DB + networks on disk).
