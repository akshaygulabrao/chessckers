# Run 24 — playout-cap randomization (KataGo PCR) on e8/d8, Gumbel S1 carried

> Run 23 proved the Gumbel S1 target learns the e8/d8 mate in ~1.5-2h wall-clock. Run 24 keeps
> that config bit-identical and adds **KataGo playout-cap randomization** (Wu 2019): per move,
> 25% chance of a FULL 800v search (noise+temperature on, training record emitted) else a FAST
> 100v search (no Dirichlet, argmax move, **no record**). Games get ~2.9× cheaper per move while
> every recorded policy row keeps 800v quality; value learning — game-count-bound under pure z —
> gets ~3× more independent outcomes per wall-clock hour. The metric is the `cc` run clock.

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `run24_V5_e8d8_c64b6_gumbelS1_pcr25` |
| Start FEN | `3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] w - - 0 1` (= run 23; compiled `board.cc kStartposFen`, unchanged) |
| Arch / optimizer | SE-ResNet v5 c64/b6 ~630K / Adam 1e-3 flat (= run 23) |
| Policy / value targets | `improved` (Gumbel S1, c_scale=0.1) / pure z (`VALUE_Q_RATIO=0`) (= run 23) |
| Init | cold (seed-0) (= run 23) |
| **PCR** | **`--pcr-full-prob=0.25 --pcr-fast-visits=100`** via bootstrap trainParams (env `PCR_FULL_PROB`/`PCR_FAST_VISITS` → `cc fresh-run` knobs). Engine defaults 1.0/100 = off; **matchParams/serverconfig untouched → gate games (128v) PCR-free**; league games are training games → PCR applies. |
| Gate / publish / EMA / league | 160g gate @ −20 + panel, publish 400, EMA 0.99, league+PFSP on (= runs 22/23) |
| Key changes | fork (uncommitted, chessckers-port): `tournament.cc/.h` PCR flags+validation+fast dicts, `game.cc/.h` per-move draw + fast search path + conditional record, `selfplay.hpp`/`chunk.hpp` per-record `ply`/`total_plies` (**moves_left density fix** — `n−i` was records-remaining; now true ply distance, byte-identical for dense games). Server (uncommitted): bootstrap env-driven PCR trainParams append; `launch_server.sh` env passthrough. Engine (uncommitted): `cc.py` `--pcr-full-prob=`/`--pcr-fast-visits=` knobs. Box scratch-build verified BUILD_OK w/ CUDA trunk pre-launch. |
| Fleet box | vast `44287736` (RTX 3060), same box; server `http://23.227.184.228:30153` |
| Started | 2026-07-20 06:32 UTC (`training_runs.created_at` clock anchor) |
| Status | **fleet auto-ended by `mate_bench` 2026-07-20 14:54 UTC** (benchmark stamped MATE @3h36m self-play basis; trainer had died 10:37; conclusion/Result + archive pending) |

## Hypothesis

PCR decouples the two heads' data economics: policy rows (rich, per-move) come only from full
800v searches — quality unchanged, ~25% of plies; value samples (one independent z per *game*)
multiply with game throughput. Expected avg search cost 0.25·800+0.75·100 = **275 visits/move ≈
2.9× cheaper**; with the serving path CPU-bound (07-18 bench: GPU 76-83%), expect roughly
**≥1.8× games/h** (~≥4.5-5k/h vs run 23's ~2.6k/h). Noise placement is the subtle win: fast
moves play the best move the small budget finds (no exploration pollution of z); full moves keep
noise+temp for policy-row diversity. **Success = wall-clock-to-convergence < run 23's ~1.5-2h**,
even if games-to-convergence rises (the bet's accepted shape). Games-needed rising >2× while
wall-clock still wins is fine; both losing = PCR hurts at this start.

## Design delta (vs run 23)

- **One functional delta: PCR on** (`0.25 / 100v`) — everything else carried bit-identical.
- Riding along: the `moves_left_target` correctness fix (verified byte-identical on dense games,
  so run-23 comparability is intact) and per-record `ply` stamping in the chunk format
  (backward-compatible: legacy dense writers fall back to `n−i`).

## Day-1 verification (pre-committed)

1. Knobs argv-verified in trainer (`--policy-target improved --value-q-ratio 0.0`) AND
   trainParams JSON in DB carries `--pcr-full-prob=0.25`,`--pcr-fast-visits=100`.
2. **Emission ratio**: records/game ÷ plies/game ∈ **[0.20, 0.30]** over ≥50 games (binomial
   p=0.25). Outside → RNG/emission bug, halt before conclusions.
3. `improved_policy` present on 100% of emitted records; one-hot read (legal-count-conditioned)
   in run-23's healthy band; `ml` spot-check = true ply gaps (not record gaps).
4. Throughput: games/h vs run 23's ~2.6k/h; GPU util (expect serving-path bound to ease).
5. **Undecided/450-ply-cap rate** vs run 23's ~0%: fast-argmax shuffle loops on weak nets are
   the known PCR risk (smoke test hit one at 8v; 100v should be mild). >5% sustained → raise
   `--pcr-fast-visits` or add small fast-move temp; log a dated bullet.

## Decision rules (pre-committed)

- **Success / conclude** — Black ≥95% of decisive games over a 1k window sustained AND
  `watch_game` forced mate → conclude; Result records **wall-clock-to-convergence vs run 23's
  ~1.5-2h** (primary) + games-to-convergence (secondary).
- **PCR-specific halt** — emission ratio outside [0.20,0.30], or `ml`/ply stamping wrong in
  chunks → halt + forensics (trainer consumption is downstream of a broken emission).
- **Throughput read** — if games/h <1.3× run 23 with healthy GPU, the serving-path bound
  swallowed the visit savings: bench (`bench_selfplay.py` recipe) before blaming PCR.
- **Abandon** — no Black-share progress by ~11k games (2× run 23's convergence count) with
  healthy chunks → halt; suspect the fast-move data path (z fidelity) before the target.
- Instrument-calibration precondition carried verbatim (template rule).
- Trainer/bridge death watch: run 23's trainer died ~step 1867 (cause in archived panes); if it
  recurs in run 24, capture pane + diagnose BEFORE restart (`cc restart-trainer` warm-resumes).

## Log

- `07-19` Staged: PCR implemented fork-side (per-move draw, fast OptionsDict child w/
  noise-epsilon=0 + temperature=0, conditional PureRecord incl. Gumbel readout skip, tree reuse
  unchanged both paths, `--pcr-full-prob=1.0` verified byte-identical no-op; local CPU build +
  live smoke: 0.3/8v → 10 records/25 plies, ml = true gaps, reuse fast-path exercised). Server
  env-driven trainParams (unset ⇒ byte-identical legacy params). `cc fresh-run` knobs wired.
  Box scratch-build **BUILD_OK** (CUDA trunk, 182/182). Run-23 archive + launch staged in
  `./run.sh` — pending user trigger.
- `07-20` **Launched** (first `./run.sh` died on the ssh-inline pkill self-match footgun —
  fixed via remote-script-file + bracketed patterns, now in the fleet-ops memory; take 2 clean).
  Clock anchor 06:32 UTC.
- `07-20` **False alarm ("kings→stones corruption") RESOLVED as NOT-A-BUG + day-1 partials.**
  `cc games` broke on a sparse chunk ('?' tokens) and record fens showed `S` stones from the
  kings-only start → misread as a rules regression predating PCR (run-23 chunks have stones
  too). Reality: **v6 Charge lawfully demotes kings → moved Stones** (spec §3C; a lone king may
  charge d=1) — PyVariant oracle reproduces 41/41 transitions of the "corrupt" run-23 chunk,
  pure-oracle random play from kings-only grows stones 20/20, merge `bafe8fe` proven exact-union
  (rules dir byte-identical run22↔run23). Stones were simply invisible-in-plain-sight on the
  full-start runs 16–22. Run 23's SUCCESS stands; no fleet action taken. REAL findings shipped
  instead: (a) adjacency-assuming tools break on sparse chunks — `cc games` '?' replay,
  `check_chunk_parity` false flags (signatures: w→w side repeat, halfmove jumps; 14 artifact /
  0 real on training.2406) → gap-aware fixes + fork-side `ply` serialization staged; (b) day-1
  partials: ~4.4k games/h @34m (**~1.7× run 23**), 6/6 promotions, sparse emission ~9.6
  records/game ≈ 0.20-0.25 of plies (band's low edge ✓), improved_policy present. Note for the
  Result: White converting via the rank-8 rule appears in early data ({r8:N} fens) — a legal
  win path the e8/d8 analysis should watch, not a symptom.

- `07-20` **`mate_bench` landed (the standardized time-to-mate benchmark) + watcher armed.**
  New `cc bench` / `engine/scripts/mate_bench.py`: first moment the trailing-1000-game window
  reaches Black ≥90% of ALL games (draws count against; decisive share also reported), crossing
  recomputed retroactively from `training_games.created_at` (exact regardless of watcher uptime),
  stamps `/workspace/chessckers/BENCH_RESULTS.jsonl` (reset-proof), then **auto-ends the run**
  (stops client + STOP-file trainer flush; server stays; `cc restart` resumes; `--max-hours` 24
  DNF bound). Metric basis finalized same day = **self-play games only** (league excluded; see
  the dilution diagnosis below). Stamped: **run 23 = 1h44m / 2,516 self-play games** (ledger's
  ≲1.6h-to-~99% read is consistent; 90% falls earlier on the ramp). **Run 24 first-crossing =
  3h36m / 12,784 self-play games** (10:09 UTC) — PCR **loses the wall-clock bet ~2.1×** (and
  ~5.1× on games) despite ~1.7× games/h. At run 24's crossing the window was B 90.0% with
  **dec 100% and 10% draws** — the crossing was DRAW-limited (PCR shuffle-loop floor, the item-5
  tripwire), not White-resistance-limited.
- `07-20` **Trainer death RECURRED** (banked death-watch): SIGKILL (`exited with -9`,
  OOM-suspect; dmesg blocked) at 10:36:54 UTC / step ~2265, right after a net upload — same
  signature as run 23 (~step 1867, also ~4h in). Two runs, same box ⇒ systematic; forensics
  captured per the pre-committed rule → box `/workspace/chessckers/run24-trainer-death-pane.txt`
  (pane scrollback + free + nvidia-smi). Net 42 (the dying upload) still gate-passed at 10:42
  (match 116: 82-71-7 vs 41) and played the whole 4h+ client-only era. Watcher's auto-end at
  14:54 = client stop (trainer already dead); server left up; `cc restart` would resume.
- `07-20` **"Post-crossing regression" DIAGNOSED — measurement dilution, NOT strength loss.**
  The all-games window slid 90.0%→83.6% after the crossing, but splitting by
  `opponent_network_id` shows frozen net 42's **pure self-play is 98.3–99.3% Black, W 0.2–0.5%,
  stationary for 4h** (run-23-endpoint quality — PCR DID fully converge). The entire White-share
  rise is league games: their completed-share roughly doubled (config `league.fraction` 0.2,
  observed 16–20% pre-death → **36%** under the stale best; games/h also ~halved 4.3k→2.3k —
  open sub-mystery, client-side), and the frozen-era league mix **farms fossil pool nets as
  White** — learner-as-White vs net 10 (561 Wwins; net 10 is from the W=100% era and cannot play
  Black) + vs net 26 (277) = ~85% of all "White wins", plus ~580 450-ply shuffle-DRAWS vs nets
  34/38 (= the 5.5% draw floor). PFSP note: opp 10 drew the most games despite wr≈1 — cached
  probs from 10:42 + the "no recent data ⇒ 50%" default persisted for 4h (main.go:379).
  **Consequence: `mate_bench` metric switched to self-play-only** (league share tracks pool
  composition, not mastery; an armed watcher polling the live all-games window would never have
  re-seen ≥90% on a fully-converged net) — both runs restamped on the corrected basis.

## Result

<staged — leave empty until the run ends.>
