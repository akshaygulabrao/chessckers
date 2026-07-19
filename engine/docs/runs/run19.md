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
| Status | **concluded 2026-07-14** (archived; superseded by run 20) |

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

- `07-13` **Monitoring stack deployed mid-run** (scripts synced; all read-only for run 19 —
  the gate untouched). Anchor cron gains: budget reallocation off saturated anchors
  (score ≥0.9 ×3 rows → 6 tripwire games, surplus → the discriminative anchor: random 6,
  seed13 34), auto-pin of a new anchor rung at full saturation, and a **plateau alarm**
  (3 rows/≥16h/<+40 Elo → `/workspace/chessckers/ALERTS.log`, `NTFY_TOPIC` opt-in push)
  with a gate stall-floor screen appended for context. `cc doctor` gains the strength-trend
  section (Elo±CI, slope/24h, PLATEAU/STALL-FLOOR flags). `cc backup` pulls jsonl/db/logs
  to Mac `telemetry/<run>/` (auto-throttled after status/doctor); `cc compare` overlays
  runs (sparklines + seed13 alignment). Verified live: plateau fires (−20 Elo/16h), gate
  screen fires (13/15, mean +31), best_net 116. The daily champs-audit cron
  (`install_monitor_crons.sh`, 04:45 → `champs_audit.jsonl`) was prepared but **not
  installed** (box-crontab edit needs the operator; runs via `./run.sh`). Gate regression
  panel (candidate must also not regress vs log-spaced past champions) implemented in
  lczero-server — lands with run 20's provision, not deployed to run 19.
- `07-14` **Floor guard added to the plateau detector before run-20 provision:** a cold
  net scores 0.0 vs seed13 for the first day → three floored rows (Δ=0 over ≥16h) would
  false-fire the alarm; `plateau_check` now returns unmeasurable (no alarm) when all
  window rows have score ≤ 0.05. Backtested on run 19's real jsonl: floored prefix →
  None, live plateau → still fires. (First finding of the plateau-detection workstream;
  detector tuning continues offline against this run's archived series.)
- `07-14` **Run concluded, fleet clean-stopped, full state archived** to
  `~/chessckers-backups/run19-fullstart-c64b6-league-20260714/` (db + 115-row matches CSV
  + networks/ 282M + games/ 362M + trainer/run1 2.0G + server/trainer/anchor-cron logs +
  crontab + serverconfig). Trees pinned for the handoff: engine `558cc44` (monitoring) +
  `0062d33` (docs), server `46b3831` (regression panel). LR was deliberately **never
  dropped** — the intact plateau is the ground-truth fixture for detector development.
- `07-14` **Post-conclusion LR-drop probe (the deferred plateau diagnosis) — VERDICT:
  the plateau is DATA-side, not optimizer-side.** Run 21 paused to free the box; the
  archived endpoint state resumed standalone at `/workspace/lr_probe/` — Adam + step
  clock from the intact `train_state.pkl` sidecar (step 7715; NB the archive's 383M
  `replay_buffer.pkl` is a **truncated `.tmp`** — the shutdown snapshot rename never
  completed — so the window was REBUILT by re-ingesting chunks `training.8776–11633.gz`
  = the exact endpoint window: 2858 games / 224,425 pos vs the endpoint's 224,400).
  `--replay-factor 0` (a frozen buffer deadlocks the ingest throttle), publish on
  timer instead of games; everything else production-identical (batch 1024, EMA 0.999,
  pure-z targets). LR **1e-3 → 3e-4** (the pre-committed drop), **+3017 steps**
  (7715→10732, ~93 min standalone ≈ +39% of the run's total optimization). Result:
  **loss on the frozen window fell hard — value 0.178→0.065 (−63%), policy
  2.525→2.457 — so the 1e-3 loss level WAS an optimizer noise floor… and strength did
  not move at all**: post-net vs endpoint best #116 (endpoint `weights.bin` is
  byte-identical to promoted #116), @128v matchParams temps via `ladder.py
  --engine`: 48–52 on the first 100g; **pooled over 300g: 136–164 (45.3%, −33±20
  Elo 1σ — base better at ~95% LOS)**. Zero-to-*negative*: harder fitting of the
  frozen window slightly hurt play, the sign memorization predicts.
  Reading: the strength-relevant content of the plateau window was already fully
  absorbed at 1e-3; the residual loss is value-head memorization of per-position
  outcome noise (pure-z targets between near-equal nets ≈ coin flips) plus the
  irreducible entropy of temp-1.0 visit targets. This **refutes "drop LR at plateau"
  as a standalone fix** (and undercuts the LR-drop decision rule's premise: the
  800v>128v headroom is real but does NOT transmit via more optimization on
  equilibrium data) and **redirects at the data/targets**: league+PFSP (run 21's
  bet, now better motivated), opening diversity, **value-from-Q / q+z blend** (the
  run-14-shelved re-test — this probe is direct evidence pure-z is the memorizable
  dead end at equilibrium), Gumbel c_scale=0.1 targets. Capacity is NOT indicted —
  the net absorbed the window easily. Caveat: the probe isolates the
  absorb-from-current-data channel; a production drop also changes future game
  generation. Artifacts: box `/workspace/lr_probe/` (nets, trainer.log, match jsons),
  mirrored into this run's archive dir.
- `07-14` **Chunk forensics on the plateau data (games 11000–11199) — the full start is
  BORN +0.50 for White; temperature exonerated as the decider.** q_white at ply 0 =
  **+0.50 ± 0.01** across games (Dirichlet barely moves it); P(final winner == sign of
  ply-0 q) = **82%, flat through ply 10** (equals the sample's 82.5% White win rate —
  up from 57% at cold start: dominance grows as conversion improves). **78% of decisive
  games never see the advantage change hands**; Black's ~18% comes from late cliff
  reversals (median ply 49, only 7% inside the 15-move temp window, 52% single-ply
  |Δq|≥0.4 — mid-game chain shots, not temp dice; per-ply volatility 0.059 in-window vs
  0.045 after). ~38% of White wins are rank-8 king-runs. Reading: the plateau's data
  pathology is a **single-narrative distribution** — every game is White converting a
  born advantage or randomly failing to; balanced two-sided positions never appear in
  training, yet seed13 beating these nets from the same start proves skill headroom
  exists that this data can't teach. Lowering temperature would *worsen* the monoculture
  (more reliable conversion, fewer Black wins). Points at **balanced-opening seeding**
  (self-play starts sampled from a q∈[−0.2,+0.2] pool, refreshed per champion) or a
  design-level start rebalance. Also measured and rejected: encoding blind spots (stack
  depth channels + chain waypoint masks are faithful), per-side search asymmetry (Black
  42.5 vs White 28 legal moves but equal visit concentration/entropy). Follow-up
  forensics (Black-learnability question): W% by era ran 83% (0–2k) → **58% (2–4k,
  Black's skills arriving)** → 70s → 78% (endpoint — White re-adapts for good); at the
  plateau **Black converts 92% of its clearly-winning positions** (34/37 at q<−0.5,
  median 18 plies) vs White 85% of q>+0.7 ones — Black's bottleneck is *access* (~19%
  of games), not conversion, and its only supply is White blunders that training keeps
  removing. **Improvement itself destroys the training signal from this start** — the
  plateau is an attractor, not a stage.

## Result

**League: works as designed; no RPS. Run: real early progress, then a genuine plateau —
concluded in it, by choice.** Endpoint 116 nets / 11,610 games / 115 gate matches
(92 promoted), best #116 (promoted on a 19-21 losing record — the stall floor in one
line); league mix verified at 21% of training games. Anchors: seed13 from −800 (floor)
to ≈ −130±20 by net ~90, then FLAT ~36h; search:3 ~+250-300; random saturated. The
anti-RPS bet passed its falsifiable reads: anchor climb at least as fast as run 18
(seed13 −168 by net #73 vs run 18's −89 at endpoint #122), and **no cycle signature**
in either the checkpoint ladder or the champion audit. What the run exposed instead:
the **gate promotes noise during a plateau** (last ~17 promotions span 81 Elo
head-to-head with best nominally LAST and rejects interleaved; ~+700 gate Elo claimed
vs ≈0 measured) — motivating run 20's regression panel — and the value head stayed
healthy throughout (800v beats 128v +241; best beats iter-200 71-29), so the plateau
is optimization- or data-side, not capacity collapse. Plateau cause **deliberately left
undiagnosed** (no LR drop): the archived series is the detector-development testbed.
*(Post-conclusion 07-14: diagnosed via the LR-drop probe — see the last Log bullet.
Verdict: **data-side.** The 1e-3 loss level was a real optimizer noise floor — 3e-4 on
the frozen endpoint window cut value loss 63% — but the fitted net played dead-even-to-
slightly-worse vs best #116 (300g: 45.3%, −33±20 Elo), so the residual loss was
strength-irrelevant (and mildly harmful) memorization; the equilibrium data itself is
what's exhausted.)*
Handoff to run 20: regression panel (server `46b3831`), monitoring/alarm stack + floor
guard (engine `558cc44`+), pre-committed decision rules in `_TEMPLATE.md`.
