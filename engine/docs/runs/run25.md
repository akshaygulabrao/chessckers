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

## Result

<staged — leave empty until both arms complete.>
