# Run 19 — league self-play (population opponents, anti-RPS)

> Run 18 (the clean `blacks_move`-fixed control) + exactly one training-loop change:
> **league self-play** — 20% of training games play the current best vs a log-spaced pool
> of past champions instead of vs itself, so counter-one-opponent strategies stop being
> rewarded in the data. Feature doc: [league-selfplay.md](league-selfplay.md).

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` (DB) | `V5_fullstart_c64b6_league` (training-run id 1, dir `run1`) |
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

## Result

<active — leave empty. Primary read: anchor trajectory + gate series vs run 18's backed-up
control curves; league mix visible in `cc status` (`league:` line) once the pool exists.>
