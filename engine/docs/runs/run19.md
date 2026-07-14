# Run 19 — league self-play (population opponents, anti-RPS)

> Run 18 (the clean `blacks_move`-fixed control) + exactly one training-loop change:
> **league self-play** — 20% of training games play the current best vs a log-spaced pool
> of past champions instead of vs itself, so counter-one-opponent strategies stop being
> rewarded in the data. Feature doc: [league-selfplay.md](league-selfplay.md).

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` (DB) | `run19_V5_fullstart_c64b6_league` (training-run id 1, dir `run1`; renamed from `V5_fullstart_c64b6_league` on 07-11) |
| Start FEN | official full start (`STARTING_FEN`, `{wm:2}`) |
| Arch | SE-ResNet gather head, c64/b6 ~630K, tag `v5` (= run 18) |
| Optimizer | Adam, lr=1e-3 (= run 18) |
| Policy / value targets | `visits` / pure z (`VALUE_Q_RATIO=0`) (= run 18) |
| Init | COLD random init (= run 18 — the A/B pairing) |
| **Gate** | **−20 lenient regression gate from t=0** (`56a368e`; run 18 ran promote-always for matches 1–25 before restoring −20 — mild confound vs run 18's early era, noted) |
| **League** | **enabled, fraction 0.2, poolSize 8** (log-spaced past champions; phases in automatically after the 2nd promotion; attribution in `training_games.opponent_network_id`) |
| Rules | v6 bottom-*d* charge (= run 18) |
| Replay buffer | unchanged (window ramp 400→4000 @α0.75, RF=8) |
| Key trees | fork `4332013` (= run 18's `ee64b19` + league sampling); server `56a368e` (league pool + attribution + −20 gate); client `93e0951` (league plumbing) |
| Fleet box | vast `44287736` (RTX 3060 datacenter — same box as run 18) |
| Started | 2026-07-11 |
| Status | active |

## Hypothesis

League opponents remove the incentive to overfit a counter to the single self-play
opponent (the RPS non-transitivity mechanism). Success read, vs run 18 as the paired
control (same config, same box, cold init):

- `cc anchor` trajectory (fixed anchors: random / search:3 / seed13) climbs **at least as
  fast** as run 18's and — the real bet — keeps climbing where run 18's flattens while
  gate-Elo keeps rising (that divergence *is* the RPS signature).
- `cc ladder` round-robins over successive champions show **fewer A>B>C>A cycles**.
- Gate promotion rate may drop somewhat (candidates can no longer pass by countering one
  opponent) — that is expected and fine; a **freeze** is not.

Failure read: anchor trajectory clearly *slower* than run 18 with no cyclicity
improvement ⇒ 20% off-distribution games cost more than the robustness buys at this
scale; try fraction 0.1 or revisit pool spacing before abandoning.

## Design delta vs run 18

- **League self-play** across the three fleet repos (fork `4332013`, server `d53527f`,
  client `93e0951`) — mechanics, wire contract, deploy-order warning in
  [league-selfplay.md](league-selfplay.md). Trainer + ccz1 format untouched; both sides'
  plies train (explicit call).
- **Gate −20 from t=0** (`56a368e`) — run 18's own mid-run restoration carried into the
  committed config. Also makes league "champions" genuinely gated.
- Nothing else: same arch, optimizer, targets, buffer, start FEN, box.

## Log

- `07-11` Run 18 concluded and frozen at **124 nets / 12,449 games / 123 matches**,
  best #122, −20 gate active and rejecting (last match #124: 18-22-0, −35, REJECT).
  **Run-18 endpoint anchors** (net #122, 20g/anchor @100 sims): random **20-0-0 (+800)**,
  search:3 **18-2-0 (+512)**, seed13 **5-5-10 (−89 [−241, +64])** — the control curve's
  final point (run 17's net ~59 read −338 vs seed13). Full state (db + matches CSV +
  networks + pgns + trainer/run1 + logs + anchor jsonl) backed up to
  `~/chessckers-backups/run18-fullstart-c64b6-fixed-20260711/` before `reset_fleet`.
- `07-11` `cc fresh-run --run-name=V5_fullstart_c64b6_league --arch=v5 --c-filters=64
  --n-blocks=6 --se-ratio=8 --value-q-ratio=0` (cold, parallelism 32) on box 44287736 —
  all 6 phases + @reboot cron OK.
- `07-11` **LIVE + verified**: server/bridge/trainer UP, `cc status` header
  `V5_fullstart_c64b6_league | gate thr -20`; box serverconfig league
  `{enabled, 0.2, 8}`; box engine binary advertises `--league-weights/--league-fraction`;
  seed13 survived the wipe at `/workspace/run13_seed/`; cold net #1 published, client
  playing on CUDA. League correctly dormant (pool empty) until the 2nd promotion —
  first league games should show in `cc status` (`league:` line) within ~1-2h.
- `07-11` **Anchor gauntlet automated on the box** (run 18 had no in-run rows because
  invocation was manual): 8-hourly cron `15 */8 * * *` runs
  `.venv/bin/python scripts/anchor_gauntlet.py` (flock-guarded; defaults 20g/100 sims =
  comparable with the run-18 endpoint row) → appends to `trainer/run1/anchor_gauntlet.jsonl`,
  log `/workspace/anchor_cron.log`. Mac-independent; survives fresh-run's cron rewrite
  (that only filters `restart_fleet.sh` lines). t≈0 baseline row kicked immediately via
  the exact cron command (validates the cron env too).
- `07-11` Run renamed in DB+cron to carry the ledger handle
  (`run19_V5_fullstart_c64b6_league`); cc commands now lead with RUN_NAME
  (strength/status reordered, gauntlet/ladder/anchor gained identity headers, anchor
  JSONL rows gain a `"run"` field); `runNN_` prefix adopted as the naming convention
  (documented in scripts/README.md).
- `07-12` First fork-played `cc ladder` (7 nets, 4 games/pair, 128v temp 1.0): range 815,
  surface read "iter-200 net beats best" — **refuted as noise**. 24 games/net ⇒ ±140 Elo
  95% CI (a 3-1 pair is p=0.31 between equal nets); only the >300-Elo gap to nets 1/67/133
  is meaningful, and the matrix has no A>B>C>A cycles (no RPS signature). The anchor cron
  settles it: net #43 ≈ iter 198 landed 07-12 16:24 at search:3 **−89** / seed13 **−636**;
  by net #73 (iter ~336) the rows read search:3 **+241** / seed13 **−168** — still climbing
  steeply through the exact stretch the ladder scored as decline. Pace vs controls: run 17
  net ~59 read seed13 −338, run 18 needed net #122 for −89; run 19 is at −168 by net #73.
  League verified live (`league:` 39/202 last hour ≈19%; DB: 20% of training_games across
  the champion pool). **Gate-Elo calibration:** `cc strength` cum (+2106) is inflated by
  construction — 40-game matches give σ≈55 Elo, so at thr −20 a *stalled* trainer still
  promotes ~64% of candidates at E[calcElo|promoted]≈+32 (last 25 matches: 80%, mean +43 —
  barely above the stall floor). De-inflated ~÷2.7 it matches the ladder's net1→best
  ~750-800. Read anchors for truth, gate cum as bookkeeping; ladder ordering claims need
  `--games ≥ 12`.
- `07-13` **Deeper-search discriminator — "does best lose to its own deeper search?" YES:**
  pinned best snapshot (≈ iter 340 / net ~#74) vs itself, 800v vs 128v, 40 games via the
  fork's own selfplay mode (`--player1.visits=800 --player2.visits=128 --no-share-trees`,
  matchParams temps, colors alternating): deep side **32-8-0 (80%, +241 Elo, LOS 99.99%)**
  — as White 19-1, as Black 13-7, 0 draws. ≈ **+91 Elo per visit-doubling**, healthy
  AZ search scaling ⇒ the value head does real search-amplifiable work; rules out both
  "policy-only memorization" and run-17-style "search anti-uses value" for the current
  stack, and corroborates the anchor-verified progress. Ops residue: first attempt over
  UCI (ladder gained `path@VISITS` per-net visits) died ~35×/40 mid-game at 800v — fine
  at 128v (yesterday's 84-game ladder was clean), isolated 800-node searches pass, cause
  unknown; `engine_uci.py` now saves engine stderr to `/tmp/uci-<net>.<visits>v.err`
  (was DEVNULL) so the next crash names itself. **Selfplay mode is the robust
  asymmetric-visits match harness; treat ladder-over-UCI ≥800v as flaky until diagnosed.**
- `07-13` **Head-to-head closes the "iter-200 beats best" question: best wins 71-29**
  (100 games @128v both sides, fork selfplay mode, matchParams temps, alternating colors;
  **+156 Elo, LOS 100%**; W 39-11, B 32-18, 0 draws). The 07-12 ladder's 3-1 for net 200
  (+71 Elo) was a 4-game artifact — a ~227-Elo swing under proper sampling, right in line
  with the ±140/net CI math. All three instruments now agree run 19's progress is real
  and correctly ordered: anchors climb through the whole 200→best stretch; best@800 beats
  best@128 32-8; best beats iter-200 71-29. **No RPS signature anywhere.** What survives
  of the 07-12 worry: gate cum-Elo is inflated (÷~2.7) and nets must never be ordered on
  ≤4-game pairs.
- `07-13` **Gate audit via `cc champs` (new tool): the gate's OWN champions, head-to-head.**
  Unlike `cc ladder` (trainer iter-checkpoints), this ladders the server's promoted .bin
  nets: field = best(#112) + log-spaced past champs {c93,c102,c106,c110,c111} + the 3
  newest rejects {r94,r107,r108}; fork @128v matchParams temps, 12g/pair, 432 games, 0
  crashes. **The last ~17 promotions (#93→#112) are FLAT: the whole field spans 81 Elo**
  (1σ ≈ ±36/net at 96 games) — best finishes nominally LAST (45%, −32), c102 nominally
  first (+49, a 1.8σ gap), and the rejects sit in the same band (46-48%). Gate verdicts in
  this stretch are coin-flips promoting at the σ≈55 stall floor: ~+700 cum Elo claimed
  over the span, ≈0 measured. The anchor cron independently agrees: seed13 flat at
  −147±40 for the last 4 rows (~24h) after the 07-12 climb; search:3 noisy-flat ~+250
  (python-MCTS anchors + fork ladder concur → no run-17-style harness split). This
  REFINES, not contradicts, the entries above: early-run progress (iter-200→best 71-29,
  800v>128v) is real — the plateau is the newest stretch only. Single-pair reversals
  (c111 10-2 over best) are within 36-pairing multiple-comparison noise; re-match at
  40-100g in fork-selfplay mode before reading them as gate contradictions. Tooling:
  `champ_ladder.py` (DB promotion history → gunzip networks/<sha> → ladder), `ladder.py`
  now takes raw .bin nets in --engine mode, `cc champs` dispatches it.

- `07-13` **Monitoring stack deployed mid-run.** Anchor budget reallocated (8-hourly cron confirmed real, no change needed). Plateau alarm added to anchor cron (writes `/workspace/chessckers/ALERTS.log`; NTFY_TOPIC opt-in on the box for push notifications). `cc doctor` slope diagnostic extended. Daily champs audit cron (`install_monitor_crons.sh`) deployed to the box via `cc ssh bash …` — appends `champs_audit.jsonl` nightly at 04:45 (12g/pair). Off-box `cc backup` added: pulls anchor_gauntlet.jsonl, champs_audit.jsonl, chessckers.db, ALERTS.log, last-2000-lines of server/trainer logs to `telemetry/run19_V5_fullstart_c64b6_league/`; auto-triggered in the background after `cc status`/`cc doctor` when >6h stale. `cc compare` adds cross-run sparkline + seed13 alignment table. Run 19's gate deliberately untouched — the monitoring is read-only this run. Gate regression panel (comparing gate Elo series vs anchor Elo for run 18 vs run 19) lands with run 20.

## Result

<active — leave empty. Primary read: anchor trajectory + gate series vs run 18's backed-up
control curves; league mix visible in `cc status` (`league:` line) once the pool exists.>
