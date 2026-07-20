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
| `cc anchor [opts]` | **current net vs FIXED anchors** (random init / alpha-beta bot / run-13 seed) → absolute strength trajectory | box |
| `cc games [opts]` | **pull a RECORDED self-play game off the box + render its real moves** | box→local |
| `cc watch [opts]` | pull the latest fleet net + watch it self-play live | box→local |
| `cc restart-trainer [LR]` | **hardened clean warm-restart** (snapshot-guarded; optionally change LR) | box |
| `cc play [opts]` | **play a human-vs-net game** against the latest fleet net | box→local |
| `cc champs [opts]` | **audit the gate's actual promoted champions + rejected candidates** (server `.bin` nets via DB; 12+ games/pair for ordering claims) — contrast with `cc ladder` (trainer checkpoints) | box |
| `cc backup` | **pull irreplaceable telemetry off the box** to `telemetry/<run>/` under the repo root | box→local |
| `cc compare [file.jsonl ...]` | **compare anchor runs** — sparklines + seed13 alignment table across runs (`cc backup` refreshes the data) | local |
| `cc bench` | **time-to-mate benchmark** — clock, current Black share, exact first ≥90%-of-trailing-1k crossing, cross-run comparison table (`BENCH_RESULTS.jsonl`); `--watch` arms the **auto-ending** watcher (stamps + stops client/trainer at crossing or `--max-hours`, server stays up); `--stop` disarms; works on archived DBs via `cc run mate_bench.py --report --db … [--stamp]` | box |
| `cc fresh-run [opts]` | **provision + launch a complete training run from scratch** (`--bench` arms the auto-ending benchmark watcher at launch) | box |
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

**RUN_NAME convention:** start it with the ledger handle (`runNN_`), e.g.
`run19_V5_fullstart_c64b6_league`. The DB training-run id is always 1 (every
fresh run wipes the DB) and means nothing across runs — the RUN_NAME is the
identity every cc command displays.

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
curve can't. Runs on the box (nets + GPU are there). Engine-mode games (the default
via `cc ladder`) run in the fork's own selfplay tournament harness — the gate
operating point; `--harness uci` is the legacy stateless driver (diagnostics only).

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

**Current vs fixed anchors** — `cc anchor` is the absolute-scale complement: ladder
and gauntlet compare against the run's own MOVING snapshots (chained Elo accumulates
noise ≈ ±55·√N over N matches, so a 20-net cumElo curve can't distinguish real +100
from a random walk). `cc anchor` plays the current net vs anchors that NEVER change
— `random` (seed-0 init of the current arch), `search:D` (net-free alpha-beta bot),
`seed13` (the run-13 warm-start seed, cross-run comparable), or any `.pt` — and
appends a JSONL row per invocation (`<run-dir>/anchor_gauntlet.jsonl`), building the
run's strength trajectory. Games are diversified by opening temperature (both
gauntlet and anchor default `--temperature 1.0 --temp-plies 20`; without it, repeat
games from the fixed start are identical and the verdict is vacuous). Keep
`--games`/`--sims` fixed across invocations so rows are comparable; the printed 95%
CI is the real precision (20 games ≈ ±150 Elo).

```
cc anchor                              # current vs random + search:3 + seed13
cc anchor --games 40                   # tighter error bars
cc anchor --current trainer/run1/iter-async-000123.pt   # measure any snapshot
```

## 4. Off-box telemetry + run comparison

### `cc backup` — pull irreplaceable telemetry

Files that live **only on the ephemeral box** and can't be reconstructed:

```
cc backup
```

Pulls to `telemetry/<RUN_NAME>/` under the repo root (created automatically; flat,
no dated subdirs — overwrite is fine because the jsonl/db are append-only):

| File | Notes |
|---|---|
| `anchor_gauntlet.jsonl` | strength time-series of record |
| `champs_audit.jsonl` | gate-champion Elo series (skipped if not yet written) |
| `chessckers.db` | full gate history (matches, promotions, rejections) |
| `ALERTS.log` | plateau alarms from the anchor cron |
| `server.log` | last 2000 lines |
| `trainer.log` | last 2000 lines |

`telemetry/` is in `.gitignore` (binary DB + large jsonls don't belong in the tree).

**Auto-trigger:** `cc status` and `cc doctor` silently spawn a background backup if
the last one is >6h old (marker file `telemetry/.last-backup`). Never blocks the
command; never fatal.

### `cc champs` — gate's real champions + rejected candidates

Unlike `cc ladder` (trainer iter-checkpoints played by Python MCTS or the fork),
`cc champs` queries the server DB for the promotion history, gunzips the `.bin` nets
from `networks/`, and runs them through a round-robin played in the fork's own
selfplay **tournament mode** at the gate's operating point (128v, matchParams temps,
per-player tree reuse) — since 07-17 literally the promotion-match harness. This is
the GATE'S own perspective — who actually got promoted, who was rejected, and whether
the champion pool has a strength order. (Stateless UCI driving — a different,
White-collapsing operating point, run22.md 07-16 — survives as `--harness uci` for
diagnostics only.)

Default 40 games/pair — the 40-game promotion-match convention (run ≤21 gate,
run 22 panel legs), ≈ ±40 Elo/net 95% CI on a full 9-net field, enough to order
the ~80-Elo fields the audit actually sees.
(4 games ≈ ±140 Elo 95% CI; 12 ≈ ±80 — run 22's 12g audit came back scrambled:
84-Elo spread with best ranked 6/8, all inside noise.) The daily cron audit
(`install_monitor_crons.sh`) inherits the default; each pair's games run
concurrently inside one engine process (`--parallelism`, default 32), ≈3–4h for
the full 9-net×40g field under fleet contention (~10s/game wall measured 07-17).

```bash
cc champs                     # log-spaced champions + 3 newest rejects, 40g/pair
cc champs --games 12          # quick look (noisy: ~±80 Elo/net 95% CI)
cc champs --pin 2 --list      # register net #2 as a PERMANENT pinned rung (pN) and exit
```

Pins (`--pin N`, 2026-07-18): a frozen copy of net #N (`networks/pins/p<N>.bin` +
`pins.json`, run-scoped) that every later audit of the run auto-includes — fixed
rungs making best-vs-pN an ABSOLUTE trajectory. This replaced the retired 8-hourly
python anchor-gauntlet **cron** (run 22 pins #2, the first promoted champion);
`cc anchor` itself remains for on-demand absolute measurements vs random/search/seed.

### `cc compare` — cross-run anchor comparison

```bash
cc compare                                        # all telemetry/*/anchor_gauntlet.jsonl
cc compare telemetry/run19/anchor_gauntlet.jsonl  # explicit file(s)
```

For each run + anchor, prints a sparkline of Elo over rows with first/last Elo and
slope (Elo/24h over the last 3 rows, assuming the default 8-hourly cron cadence).
Then an alignment table for the discriminative anchor `seed13`: rows = row index,
columns = runs — so two runs can be eyeballed at matched progress.

Robust to missing fields and short files (short series → partial table).

### Anchor cron + ALERTS.log + NTFY_TOPIC

The anchor cron (`15 */8 * * *` on the box, installed by `cc fresh-run`) runs
`anchor_gauntlet.py` every 8 hours and appends a JSONL row to
`trainer/run1/anchor_gauntlet.jsonl`. If the discriminative anchor (`seed13`) shows
no improvement over consecutive rows, it writes an alarm line to
`/workspace/chessckers/ALERTS.log`.

**Push notifications (opt-in):** set `NTFY_TOPIC=<your-topic>` in the box env
(e.g. add to `/etc/environment` or the cron environment) to forward alarm lines to
`https://ntfy.sh/<your-topic>`. The cron reads this variable at runtime; no code
changes needed.

### `install_monitor_crons.sh` — deploy monitoring crons to the box

```bash
cc ssh bash /workspace/chessckers/engine/scripts/install_monitor_crons.sh
```

Adds to root's crontab (idempotent — grep-guarded, no duplicates):

- **04:45 daily — champs audit:**
  `champ_ladder.py --jsonl trainer/run1/champs_audit.jsonl`
  → appends a JSONL row to `champs_audit.jsonl`, log at `/workspace/champs_cron.log`.
  Games/pair = the script's default (40); pin a smaller `--games` in the cron line
  if late-run game lengths make the nightly window too long.

The anchor 8-hourly cron is installed by `cc fresh-run`; this script adds the
slower daily audit that is too expensive for the 8-hourly cadence.

## 5. Playing the net

Play a human-vs-net game from any FEN — you pick from a numbered legal-move menu
(no hand-typing cadence/deploy UCI), the net replies via the REAL lc0 fork
(default whenever a fork build is found; the .pt auto-exports to .bin), driven
with full game history so it keeps its tree between moves — the production
operating point (since 07-17; the old stateless driving was White-collapsing).
`--mcts` forces the Python PUCT opponent, which also renders WDL eval + top
lines each ply:

```bash
cc play --color black                # play the LIVE fleet net (pulls it, fork @128v)
.venv/bin/python scripts/play_net.py --color white --visits 128
.venv/bin/python scripts/play_net.py "<FEN>" --weights X.pt --color black
.venv/bin/python scripts/play_net.py --mcts --sims 200 --device mps  # legacy opponent
```

`--color` is the side YOU play (default black = the towers); at your turn `u`
undoes your last move and `q` quits. Don't raise `--visits` to 800 — the fork's
UCI mode hard-crashes there (run22.md). To watch the net play ITSELF instead,
use `watch_game.py` / `cc watch`.

## 6. Script index

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
| `anchor_gauntlet.py` | box | **current net vs fixed anchors** — absolute strength trajectory (JSONL history) |
| `play_net.py` | local | **human-vs-net** play from any FEN (numbered move menu) |
| `champ_ladder.py` | box | ladder the gate's promoted champions + rejects (server `.bin` nets from DB) |
| `install_monitor_crons.sh` | box | idempotent: add daily champs audit cron to root's crontab |
| `../lczero-server/scripts/new_run.sh` | box | guarded fresh-run setup (sets start FEN) |

## 7. Gotchas (hard-won)

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
