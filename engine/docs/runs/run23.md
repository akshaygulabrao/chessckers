# Run 23 — Gumbel Stage-1 re-test on e8/d8 KK-vs-K (c_scale=0.1, cold)

> Run 15's Gumbel halt verdict was found partially void at the run-22 postmortem (its match
> evidence ran through the blacks_move-inverted harness; see run15.md retro-caveat). Run 23
> re-tests Gumbel Stage-1 where ground truth is known and verification is fast: the runs-5/6
> e8/d8 endgame, which Black provably wins and a c16/b1 net once learned outright.

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `run23_V5_e8d8_c64b6_gumbelS1` |
| Start FEN | `3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] w - - 0 1` — two kk towers d8/e8 vs K on e1, White to move, no `{wm:2}` (identical to runs 5/6; compiled into fork `board.cc kStartposFen`) |
| Arch | SE-ResNet gather head, c_filters=64, n_blocks=6, ~630K params, tag `v5` (unchanged from run 22; runs 5/6 used c16/b1 — capacity is deliberately not a variable here) |
| Optimizer | Adam, lr=1e-3 (flat) |
| **Policy target** | **`improved`** — Gumbel Stage-1 `softmax(logP + σ(completedQ))` root readout, **c_scale=0.1** (mctx reference; run 15 ran 1.0 — the banked run-16 requirement #1) |
| **Value target** | **pure z** (`VALUE_Q_RATIO=0`, the Stage-1 pairing; e8/d8 games are short and decisive → low label noise) |
| Init | **cold** (seed-0 random) — cleanest "does the target learn" read; run 15 warm-started, runs 5/6 were cold |
| Gate | run-22 config carried: 160g main gate @ thr −20 + regression panel, publish 400, EMA 0.99, league+PFSP enabled (watch for degenerate pool behavior on a fast-converging start) |
| Key commits / branch | branch reunification 2026-07-19: engine/fork/server `ctl/pre-gumbel-run16` merged back to main lines; Gumbel restored (engine revert of `5615196`, fork merge carries `79702a7`, server revert of `dcbe1df`); c_scale 1→0.1; `board.cc` → e8/d8 |
| Fleet box | vast id `44287736` (RTX 3060) — server `http://23.227.184.228:30153` |
| Started | 2026-07-19 (19:31 PDT = 2026-07-20 02:31:42 UTC, `training_runs.created_at` — the `cc` clock anchor) |
| Status | **concluded 2026-07-19 (~21:45 PDT) — SUCCESS** → run 24 |

## Hypothesis

Run 15's chunk-verified mechanism (c_scale=1 ⇒ one-hot on argmax-completedQ) is fixed by
c_scale=0.1; its *strength* collapse was never validly measured. If the Gumbel improved-policy
target is sound, a cold c64/b6 net should learn Black's forced win on e8/d8 **at least as
readily as the visit-target runs 5/6 did** — success = sustained ~all-Black self-play outcomes
with the mate verifiable in `watch_game`, under a healthy gate. Failure in the run-15 signature
(gate freeze + one-hot-crowned noise + champs-pin regression) is now observable with valid
instruments and would re-convict the target itself rather than the harness.

## Design delta (vs run 22)

- **Start FEN** → e8/d8 KK-vs-K (fork `board.cc`; fork rebuild at provision). Verified via
  `PyVariantClient().new_game()` at stage time.
- **Policy target** visits → `improved` (Stage-1 readout restored by the branch reunification;
  `--policy-target=improved`), **c_scale 1→0.1** (fork emission constant).
- **Value target** q0.5 → pure z (`--value-q-ratio=0`).
- **Init** warm → cold.
- Carried unchanged: v5 c64/b6, Adam 1e-3, gate/panel/publish/EMA/league config, P32.
- Three deltas vs run 22 by design — this is a re-run of the run-15 *experiment* in a verifiable
  setting, not a controlled comparison against run 22. Controls: runs 5/6 (visit-target baseline,
  same start) and run 15's records (c_scale=1, full start).

## Log

- `07-19` Staged: run-22 postmortem + archive plan written; branch reunification + c_scale=0.1 +
  board.cc e8/d8 prepared; launch pending `./run.sh` (archive → `cc fresh-run`).
- `07-19` **Launched + day-1 verification PASSED.** Archive completed (WAL caveat → run22.md),
  fresh-run provisioned box `44287736`, knobs argv-verified in the live trainer
  (`--policy-target improved --value-q-ratio 0.0`). First hours: ~129 games @ ~1,190 games/h,
  **W 97.1 / B 2.9 / d 0** (matches run 6's expected cold start — early games all White),
  vsign 0.997 (degenerate on the W-monoculture, not yet informative), trainer 0.19 steps/s.
  **Emission verified**: `improved_policy` present in 23/23 records of the newest chunk and
  100% of 643 pooled records. **One-hot read (the pre-committed check), conditioned on legal
  count**: ≤2 legal → argmax mass 1.000 (forced, legit); 3–6 legal → 0.942 mean, 80% >0.9;
  **≥7 legal → 0.766 mean, only 42% >0.9** — real mass spread survives on wide positions, NOT
  the run-15 flat one-hot. Improved-vs-visits argmax agreement 27% — expected on a cold net
  (visits ≈ Dirichlet+temp noise); this number should RISE as value un-blinds. Watch: Black
  share vs the runs-5/6 curve; agreement trend; re-run the one-hot read at first promotion.
  ALERTS.log cleared post-archive (stale run-22 python-gauntlet plateau lines); run_doctor
  now filters alerts to the current run.
- `07-19` **Mate found within ~the first hour** (user wall-clock observation; runs 5/6 took
  ~overnight). Caveat: wall-clock is throughput-confounded (this fleet runs P32 + tree reuse +
  publish-400 at ~2k games/h) — the Gumbel-attributable read is **games-domain**: Black share
  2.9% → **44%** of the newest-2000 window by ~2.8k games (run 5 needed ~38k games to converge).
  Verified live @ step 1530: balance W51/B44/d4 and climbing, len p50 ↑+31, **6/6 gate promotions**
  (best #7, last gate 81-79-0 +4), latest `cc games` replay = clean Black mate in 42 plies.
  No PCR / no visits change in the tree — the S1 target (+pure z, cold) is the delta vs runs 5/6.
  Pending per decision rules: one-hot re-read now that promotions exist (banked req #2), Black
  ≥95% over a 1k window + `watch_game` forced mate before concluding SUCCESS.

## Decision rules (pre-committed)

- **Success / conclude** — Black share of decisive self-play games ≥95% sustained over a 1k-game
  window AND `watch_game` shows a clean forced mate from the start FEN → conclude SUCCESS, write
  Result with games-to-convergence AND wall-clock-to-convergence (the `cc` run clock; primary
  metric going forward — run 24 PCR bets on wall-clock, not games) vs the runs-5/6 reference.
- **Freeze watch (banked run-16 req #2)** — ≥5 consecutive gate rejections → run the chunk-level
  one-hot read FIRST (fraction of plies with improved-policy argmax mass >0.9, σ magnitude)
  before any strength conclusion; champs-pin regression is the distill-below-teacher tripwire.
- **Instrument calibration precondition** — carried verbatim from the template (a strength
  instrument backs a decision only if harness-calibrated; two instruments disagreeing = alarm).
- **Abandon** — no Black-share progress after ~2× the runs-5/6 convergence game count with
  healthy chunks (improved_policy present, not one-hot) → halt + forensics; suspect trainer
  consumption before search.
- Anchor/seed13 rules do NOT apply (different position; python gauntlet retired) — strength
  reads are gate + champs pins only.

## Result

**SUCCESS — the Gumbel S1 improved-policy target is vindicated at c_scale=0.1.** Concluded
2026-07-19 ~21:45 PDT at clock **2h06m** / **5,455 games** / trainer step ~1867: Black **99%**
of the newest-2000 window (criterion ≥95%/1k cleared), 8/8 gate promotions (best #9), zero
draws, `cc games` replays showing clean tower mates. The `watch_game` forced-mate verification
was waived by user pivot (run-22 style); archived weights allow it retroactively.

- **Wall-clock-to-converge ≲1.6h** (the new primary metric): the trainer+bridge **died ~step
  1867** (~30-40 min before conclusion; cause undiagnosed — pane scrollbacks + dmesg preserved
  in the archive) and Black share was already ~99% on clients running net #9, so learning
  finished *before* the death. Reference for run 24: call it **~1.5-2h**.
- **Games-domain ≈8× vs the visit-target baseline**: converged ≲5k games vs run 5's ~38k (same
  start; confounds: arch c64/b6 vs c48/b5, 2026-07 fleet config — publish 400/EMA 0.99/160g
  gate/tree reuse — so 8× is indicative, not controlled).
- **Run-15 failure signature absent on every axis**: no gate freeze (8/8 promoted), no one-hot
  (day-1 read: ≥7-legal argmax mass 0.766, 42%>0.9), no distill-below-teacher tripwire. The
  c_scale=1 one-hot arithmetic remains the sole confirmed run-15 defect.
- Mechanism read (day-1): improved-vs-visits argmax agreement 27% on the cold net — the target
  taught Q-discoveries while visit counts were still Dirichlet noise; exactly the S1 pitch.
- Throughput context: ~2.6k games/h avg (P32 + tree reuse + publish 400 fleet).
- Open items carried out of the run: trainer/bridge death forensics (archived panes), the
  promotion-time one-hot re-read (waived), Stage-2 Sequential Halving (still unported).

Successor: **run 24** (`run24.md`) — KataGo playout-cap randomization (PCR 0.25/100v) on the
identical config; the bet is wall-clock-to-mate ≲ run 23's on ~2.9×-cheaper games. Archive:
`~/chessckers-backups/run23-e8d8-c64b6-gumbelS1-20260720/` (DB WAL-checkpointed per the run-22
lesson, networks, games+pgns, trainer/run1, telemetry, tmux panes + dmesg).
