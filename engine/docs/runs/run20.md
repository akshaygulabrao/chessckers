# Run 20 — gate regression panel (promote only if no regression vs past champions)

> Run 19 (league self-play baseline) + exactly one training-loop change: the **gate
> regression panel** — a candidate must beat the current best (40g, thr −20, unchanged)
> AND not regress vs 2 log-spaced past champions (20g legs each, thr −50). Run 19 showed
> the single-opponent gate promotes coin-flips at the stall floor (last ~17 promotions
> spanned 81 Elo head-to-head; #116 promoted on a 19-21 record); the panel is the
> promotion-decision half of the anti-RPS/anti-noise design (league = the data half).

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` (DB) | `run20_V5_fullstart_c64b6_panel` (training-run id 1, dir `run1`) |
| Start FEN | official full start (`STARTING_FEN`, `{wm:2}`) (= run 19) |
| Arch | SE-ResNet gather head, c64/b6 ~630K, tag `v5` (= runs 18/19) |
| Optimizer | Adam, lr=1e-3 (= run 19; LR-drop rule pre-committed below) |
| Policy / value targets | `visits` / pure z (`VALUE_Q_RATIO=0`) (= run 19) |
| Init | COLD random init (= runs 18/19 pairing) |
| **Gate** | −20 lenient gate + **regression panel** (`matches.panel {enabled, opponents 2, games 4×5=20, threshold −50}`; legs TestOnly+PanelParentID, invisible to elo/league/strength consumers; empty pool → legacy gate) |
| League | enabled, fraction 0.2, poolSize 8 (= run 19) |
| Rules | v6 bottom-*d* charge (= runs 18/19) |
| Replay buffer | unchanged (window ramp 400→4000 @α0.75, RF=8) |
| Key trees | engine `558cc44`+`0062d33` (monitoring/alarm stack + floor guard); server `46b3831` (panel); fork `4332013` (= run 19); client `93e0951` (= run 19) |
| Fleet box | vast `44287736` (RTX 3060 — same box as runs 18/19) |
| Started | 2026-07-14 |
| Status | active |

## Hypothesis

During healthy climb the panel is nearly invisible (a genuinely stronger net doesn't
regress vs older champions); during a stall it slashes the promotion rate (run 19's
stall-floor: 92/115 promoted; a coin-flip candidate must now also not lose ≥50 Elo to
2 champions). Success reads: (a) promotion rate drops sharply *only* when anchors are
flat (the panel becomes a plateau signal itself); (b) champion audits (`cc champs`,
daily jsonl) show a tighter, better-ordered field than run 19's 81-Elo scramble with
best nominally last; (c) anchor trajectory no slower than run 19's (panel costs ~80
gate games/candidate vs 40 — throughput hit must not dominate). Failure read: gate
freeze (many consecutive panel rejections while anchors still climb) → panel threshold
−50 too tight or legs too small; loosen before abandoning.

## Design delta vs run 19

- **Gate regression panel** (server `46b3831`): panel legs created on upload alongside
  the main match; promotion deferred until all legs land; any leg calcElo ≤ −50 rejects
  with the offending champion named in the `[gate]` log line.
- **Monitoring/alarm stack** (engine `558cc44`, deployed late run 19, first full run
  here): anchor budget reallocation + saturation auto-pin + plateau alarm (with cold-
  start floor guard) + gate stall-floor screen → `ALERTS.log`/NTFY; `cc doctor`
  strength-trend; daily `cc champs` audit jsonl; off-box `cc backup`/`cc compare`.
- Nothing else: same arch, optimizer, targets, buffer, start FEN, league config, box.

## Log

- `07-14` Run 19 concluded in its (deliberately preserved) plateau and archived to
  `~/chessckers-backups/run19-fullstart-c64b6-league-20260714/`; fleet clean-stopped.
- `07-14` `cc fresh-run --run-name=run20_V5_fullstart_c64b6_panel --arch=v5
  --c-filters=64 --n-blocks=6 --se-ratio=8 --value-q-ratio=0 --parallelism=32` (cold).
- `07-14` **PFSP league weighting landed in-tree** (engine/client/server; design in
  `league-selfplay.md`): pool sampling weighted by live per-opponent win rate,
  `f_hard=(1−wr)²` + 20% uniform floor, probs cached per (best, pool). **OFF for run
  20** — the box serverconfig predates `league.pfsp`; auto-on at next provision. New
  `training_games.result`/`learner_is_black` columns start populating whenever the
  new client deploys (deploy engine BEFORE client — old engines fatal on
  `--league-probs`).
- `07-14` **LIVE + verified** (07:39 box time): server/bridge/trainer/client UP, header
  `run20_V5_fullstart_c64b6_panel | gate thr -20`; box serverconfig carries
  `panel {enabled, 2, 4, -50}`; net #1 bootstrap-promoted via the legacy path (correct —
  panel legs only appear once the champion pool is non-empty, ~2nd-3rd promotion);
  anchor cron survived the crontab rewrite; @reboot cron carries the run-20 env; first
  game chunk landed <1 min after client start. Champs-audit cron NOT yet installed
  (operator step: `./run.sh` → `install_monitor_crons.sh`). Expect seed13 floored
  (−800, score 0) for the first ~day — the plateau alarm's floor guard keeps it quiet;
  first meaningful slope reads once seed13 lifts off the floor.

## Decision rules (pre-committed)

- **LR drop** — trigger: `seed13` gains < +40 Elo over 3 consecutive anchor rows (~24h)
  with ingest healthy (step-rate normal, buffer not starved) and the anchor NOT floored
  (score > 0.05 — cold-start floor is unmeasurable, not flat). Confirm headroom via
  800v-vs-128v (≥40g, fork selfplay mode; real scaling ⇒ optimization-limited). If
  confirmed: `cc restart-trainer 0.0003` (×0.3). Re-arm after each drop.
- **Plateau definition** — `seed13` flat (95% CI overlapping zero gain) for ≥3 rows AND
  the daily `cc champs` audit spanning <100 Elo across the same stretch. Both
  instruments, not either.
- **RPS check cadence** — daily champs audit jsonl; A>B>C>A cycles beyond
  multiple-comparison noise over ≥3 consecutive audits = RPS signature → revisit league
  fraction/pool spacing.
- **Panel health** — promotion rate collapsing (≥5 consecutive panel rejections) while
  anchors still climb = panel too tight → raise `panel.threshold` toward −20 or shrink
  legs; log any change here with a dated bullet.
- **Anchor rotation** — auto-pin fires at saturation (cron does this); record the pin +
  row index in the Log when it happens.
- **Abandon / pivot** — anchor trajectory clearly slower than run 19's archived curve at
  matched net-count for ≥48h with the panel exonerated (promotion rate normal) → Result
  entry + successor run doc.

## Result

<active — primary reads: anchor trajectory vs run 19's archived jsonl (`cc compare`),
promotion-rate-vs-anchor-slope coupling (the panel's signature), daily champs-audit
spread/best_rank trend.>
