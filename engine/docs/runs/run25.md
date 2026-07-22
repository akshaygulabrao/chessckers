# Run 25 — mate_bench A/B: Gumbel vs Gumbel+PCR, 5 trials each (noise check)

> Run 23 (Gumbel S1) crossed at 1h44m; run 24 (identical + PCR 0.25/100v) at 3h36m — a ~2.1×
> wall-clock loss, but each is n=1. Run 25 reruns BOTH configs as 5-trial `mate_bench`
> experiments (distinct trainer seed per trial, seeds 0–4 per arm) to decide whether the PCR
> penalty is real or run-to-run randomness. Not a training run in the usual sense — the two
> arms are exact reruns of runs 23/24; this doc is the experiment ledger.

## Identity

| Field | Value |
|---|---|
| Arm A | `run25a_e8d8_gumbelS1_bench` — bit-identical run-23 config (v5 c64/b6, Adam 1e-3, `improved` c_scale=0.1, pure z, cold, 160g gate @ −20 + panel, publish 400, EMA 0.99, league+PFSP, p32), NO PCR |
| Arm B | `run25b_e8d8_gumbelS1_pcr25_bench` — arm A + `--pcr-full-prob=0.25 --pcr-fast-visits=100` (= run 24) |
| Start FEN | `3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] w - - 0 1` (compiled `board.cc`, unchanged) |
| Trials | 5 per arm, trainer seeds 0–4 (paired across arms), `mate_bench` defaults: thr 0.90 of ALL over window 1000, self-play-only basis, `--max-hours 10` DNF bound (24h default tightened — worst observed crossing is 3h36m) |
| Metric | wall-clock to first trailing-1000 self-play window ≥90% Black (retro-exact from `training_games.created_at`); games-to-crossing secondary |
| Orchestration | ONE trigger: `run.sh` archives run 24 → `cc fresh-run` arm A → arms box-side `bench_chain.sh` in tmux `cc-bench`: `mate_bench --trials 5` (arm A) → rewrite @reboot cron to arm B env → reset_fleet → restart_fleet → `mate_bench --trials 5` (arm B). Per-trial DBs → `/workspace/chessckers/bench_trials/`; stamps + per-arm summaries → `BENCH_RESULTS.jsonl` (reset-proof) |
| Death-watch | chain runs a trainer babysitter (2/2 prior runs: SIGKILL ~4h in): client-alive + trainer-dead for 2×120s → forensics to `/workspace/chessckers/trainer-death-*.txt` → warm restart via `restart_fleet.sh` (weights.pt warm-resume; DB/run-clock survive; post-death replay-sampling seed reverts to base — logged in `trainer-restarts.log`, acceptable perturbation vs a 10h DNF) |
| Fleet box | vast `44287736` (RTX 3060), server `http://23.227.184.228:30153` |
| Prior n=1 points (context, kept in the ledger table) | run 23: 1h44m / 2,516 games (seed 0) · run 24: 3h36m / 12,784 games (seed 0), draw-limited crossing |

## Hypothesis / decision rules (pre-committed)

- **H0 (noise):** arm A and arm B wall-clock distributions overlap heavily (medians within each
  other's min–max spread). **H1 (real):** arm B's median clearly exceeds arm A's with
  non-overlapping or barely-overlapping spreads — run 24's 2.1× was signal; PCR (at 0.25/100v)
  hurts time-to-convergence on this start despite ~1.7× games/h.
- Read medians first (mate_bench summary lines), then per-trial spreads; 5v5 is too small for
  formal tests — a ≥2× median gap with disjoint ranges is decisive, anything less is "PCR
  roughly neutral-to-worse, not resolved at n=5".
- Secondary reads: games-to-crossing (expect arm B ≫ arm A regardless — the accepted bet
  shape), draw share at crossing (arm B's shuffle-loop floor), games/h.
- DNF (10h) counts against its arm; a babysitter restart mid-trial is logged, the trial stands
  (wall-clock includes the outage — real-world cost).
- Trial-1 sanity anchors: arm A trial 1 (seed 0) should reproduce ~run 23, arm B trial 1
  (seed 0) ~run 24 — gross divergence = harness/config drift, halt and diagnose before
  trusting later trials (instrument-calibration rule).

## Log

- `07-20` Staged: run-24 archive + arm A `cc fresh-run` + `bench_chain.sh` arming in `run.sh`
  — pending user trigger. Run 24 concluded (see run24.md Result).
- `07-20` Launched 18:26 UTC. **Trial A1 (seed 0): MATE @ 1h18m / 2,837 self-play games**
  (window@cross B 90.0%, dec 99.1%, D 9.2%) — faster than run 23's 1h44m on the same config
  +seed, i.e. run-to-run spread is real and material to the A/B question.
- `07-20` **OOM massacre → driver redesign.** ~19:45, during the trial-A2 reset/relaunch, the
  container's cgroup OOM killer killed the tmux SERVER — fleet, watcher, and the tmux-hosted
  `bench_chain.sh` driver died as one process tree (box `memory.events`: oom 202 /
  **oom_kill 117** lifetime — also the prime suspect for the run-23/24 trainer SIGKILLs;
  page-cache-accounted cgroup + bulk-delete dirty-page flood at reset is the likely trigger).
  Trial A1 was already banked (stamp + trial1.db). Fix: **`bench_resume.sh`** — a */5 cron
  driver (survives any process-tree kill) that reconstructs state from per-arm trial-stamp
  counts in `BENCH_RESULTS.jsonl`, syncs the @reboot cron line to arm+seed, fresh-launches or
  warm-resumes as appropriate, babysits trainer-only AND full-fleet death, and runs
  `mate_bench --trials <remaining>` (patched to accept `--trials 1`). `reset_fleet.sh` now
  `sync`s after the wipe. Redeploy staged in `run.sh`.

- `07-20` **Resume-driver shakedown: two more bugs, both mine, both fixed.** (a) Fresh-launch
  race: the driver started `mate_bench` right after `restart_fleet` returned, but bootstrap
  creates the DB asynchronously → `mate_bench` hard-exited (rc=1, 20:25); harmless to the
  metric (crossings are retro-exact from `training_games.created_at` — trial A2's clock runs
  from 20:25 regardless) and self-healing by design, but now fixed with a wait-for-run-row
  loop. (b) **flock-fd inheritance wedge**: cron's `flock` holds the lock on fd 3, inherited
  by every child — `restart_fleet`'s `tmux new-session` DAEMONIZED a tmux server that kept
  fd 3 open forever, so the lock stayed held after the driver exited and all later cron fires
  silently no-oped (the fleet ran trial A2 with NO watcher, 20:25→20:45). Fix: every
  daemon-spawning child (`reset_fleet`/`restart_fleet`/`mate_bench`) runs with `3>&-`;
  recovery was `rm` of the lock file (relock on a fresh inode). Watcher re-armed ~20:50;
  trial A2 unaffected (memguard live from 20:25: dirty≈0, oom_kill flat at 117).

- `07-21/22` **ENGINE MEMORY-LEAK LIVELOCK poisons trial B2 (measurement caveat: B2's DNF is
  infrastructure-censored, NOT a learning verdict).** During B2's gate phase (~02:00 UTC on,
  candidates 21/22), the 128v match engine leaks anon memory at **~45GiB/min** — a clean
  sawtooth 5G→190G→OOM-kill→restart every ~4 min (`memguard.jsonl`; lifetime oom_kill
  117→247+). memguard's victim steering worked (engine dies, tmux/fleet survive, client
  restarts the assignment) but the assignment is deterministic → **livelock**: generation
  STOPPED ~2.6h, trainer idle, match games trickling ~1-3/min between kills. Crucially B2's
  self-play trend was **B 23.2% and climbing** (blocks: 4.0%→23.2%) when learning froze — its
  10h DNF must be scored as *censored-by-bug*, unlike A2's genuine draw-equilibrium DNF.
  Monitor blind spot exposed: kills reset the FROZEN counter (moving-but-livelocked); memguard
  ALERTS + `cc doctor` caught it. Decision: NO mid-trial surgery — let the 10h bound stamp
  (~07:20 UTC), reset clears the poisoned assignment, B3 starts fresh. Repro preserved:
  `/workspace/chessckers/bug-repro-oomleak/` (both nets + exact selfplay match args) — leak
  suspect: pathological capture-chain/edge explosion on a shuffle position at 128v
  `--no-share-trees`. **Post-bench TODO: repro locally, fix fork-side; consider match-engine
  RLIMIT_AS + gate-timeout guard.**

## Result

<staged — leave empty until both arms complete.>
