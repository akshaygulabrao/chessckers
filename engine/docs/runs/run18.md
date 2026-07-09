# Run 18 — honest matches: the `blacks_move` driver fix (run 17 continued warm)

> Run 17's forensics found that `selfplay/game.cc` toggled `blacks_move` per ply, assuming strict
> alternation. The `{wm:2}` opening double-move desyncs the toggle from ply 1, so in every TWO-player
> game each side's moves were chosen by the **opponent's** engine and results were credited to the
> assigned colors — inverting the engine attribution of every full-start match since run 10.
> Training self-play (one net in both slots) was unaffected. Run 18 = run 17's training state,
> continued warm, with exactly one change: the fixed driver (fork `ee64b19`).

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` (DB) | `V5_fullstart_c64b6_cold_nogate_fixed` (training-run id 1, dir `run1`); the run NUMBER is this ledger's handle |
| Start FEN | official full start (`STARTING_FEN`, `{wm:2}`) |
| Arch | SE-ResNet gather head, c64/b6 ~630K, tag `v5` |
| Optimizer | Adam, lr=1e-3 (unchanged — exonerated by run 17) |
| Policy / value targets | `visits` / pure z (`VALUE_Q_RATIO=0`) — unchanged |
| **Init** | **COLD random init** — originally planned as a warm continuation of run 17, but the KR box (44017141) flapped offline twice post-deploy and was destroyed before a state backup landed; run-17's DB/nets/buffer/anchor-JSONL died with it (its measured numbers survive in run17.md). Net effect: run 18 is the CLEAN experiment — run 17's exact config, fixed driver, from game 0. |
| **Gate** | **promote-always** (threshold −9999, set post-provision); matches = pure measurement series — **honest from the very first gate row** (no negated era in this DB) |
| Rules | v6 bottom-*d* charge |
| Replay buffer | unchanged (window ramp 400→4000 @α0.75, RF=8) |
| Key trees | fork **`ee64b19`** (= run-17's `45349d9` + `blacks_move` fix); engine `5615196`+docs; server `dcbe1df` — otherwise identical to run 17 |
| Fleet box | vast `44287736` (RTX 3060, **datacenter** — replaced the flaky KR consumer host) |
| Started | 2026-07-09 |
| Status | **active** |

## Hypothesis

With attribution honest, the measurement layer should finally agree with the trainer:

- **Success read:** the promote-always cumElo series turns **positive**, slope ≈ the anchor-implied
  +10–20/net; `cc anchor` trajectory keeps climbing (run 17: +191 → +301 → +436 vs random at nets
  ~16/33/59) and closes the seed13 gap (−338 at net ~59).
- **Failure read:** cumElo still bleeds with the fixed driver ⇒ something real remains (then the
  optimizer/buffer hypotheses come back to the table).
- The run-14/15 "diseases" (gate freeze, dose-response) should NOT recur — they were this artifact.

## Design delta vs run 17

- Fork `ee64b19`: `SelfPlayGame::Play` recomputes `blacks_move = tree_[0]->IsBlackToMove()` each
  iteration (per-ply toggle deleted). Also transitively corrects resign adjudication + max_eval
  bookkeeping (same consumer; resign is dormant at resignpct=0).
- **Nothing else.** Same weights, buffer, config, box, gate setting.

## Verification battery (must pass with the new binary before trusting anything)

1. Same-net `--player1.visits=800 --player2.visits=1` (100g): P1 must score **~95%+** (was "3.5%" = swapped).
2. Current-net vs net #1 cold init @128v (400g): P1 must be **strongly positive** (was "+2−397").
3. Same-net null @128v (400g): must stay **≈50/50** with ~79% White-by-color.

## Log

- `07-08` Bug found + fix committed (fork `ee64b19`) during run-17 forensics; full audit trail in
  [run17.md](run17.md) 07-08 Log entry. Search/eval/export/training-data all exonerated by direct
  test; only the driver was wrong.
- `07-08` Deploy blocked ~2h: box 44017141 host-offline (vast `intended: running / actual:
  offline`). Came back WITHOUT a reboot (tmux sessions + `/tmp` artifacts survived — network/host
  blip, fleet never stopped; nets kept publishing to **#89**).
- `07-09 00:33 UTC` **DEPLOYED.** Pushed `game.cc`, stopped client + engine, `ninja` relink (binary
  mtime 00:32:55; banner still says Jul 6 — build-date string bakes at meson configure, ignore),
  relaunched client (verified via `/proc/<engine>/exe` → the rebuilt binary). Server + trainer
  untouched throughout (warm, same tmux). **Honest-era boundary: deployed after net #89** — match
  rows for cand ≥ #90 are honest; rows #2–#89 read negated. Verification battery launched on the
  SAME frozen pair as the broken-era tests (`/tmp/nullmatch/net.bin` = net ~59, `/tmp/net1.bin`) —
  perfect A/B, only the driver changed: (1) same-net v800-vs-v1 100g, (2) net59-vs-net1 400g @128v,
  (3) same-net null 400g. Expected: ~95%+ / strongly positive / ~50-50. Results below when they land.

- `07-09` **Box replaced → run 18 restarted COLD on vast `44287736`.** The KR host flapped offline a
  second time ~1h after the deploy; user destroyed it and provisioned a datacenter instance. Run-17
  on-box state lost (last lineage point: net #89, chain-true ≈ +1036 ≈ +12/net; anchor rows for nets
  16/33/59 preserved in run17.md). Relaunched via `cc fresh-run
  --run-name=V5_fullstart_c64b6_cold_nogate_fixed --arch=v5 --c-filters=64 --n-blocks=6 --se-ratio=8
  --value-q-ratio=0` (cold, no `--base`). Direct A/B available: this run's honest cumElo/anchor curve
  vs run-17's bias-corrected one (same config, same games/net cadence — search:3 crossing landed
  between nets 16–33 there). Post-provision steps: threshold −9999 + server restart; re-ship run-13
  seed to `/workspace/run13_seed/weights.pt` (anchor `seed13` leg); verify fix in box `game.cc`,
  trainer argv (cold, q=0), first-chunk sanity ({wm:2} start, pure z, wm2 sign).
- `07-09` **LIVE + verified on 44287736.** All 6 fresh-run phases + cron OK; fix confirmed in box
  `game.cc` + binary; gate thr **−9999** applied (server restarted pre-games); seed13 shipped for
  anchors. First chunk verified: start FEN = official full start `{wm:2}`, `wdl_target` pure one-hot
  ±1 z, **wm2 same-mover sign intact** (ply0 W−L=−0.028 ≈ ply1 −0.031, flip at ply2), no
  `improved_policy`. First game: Black win in 13 plies (cold-net mandate chains — run 17's opener was
  a 23-ply Black win). `cc doctor` header: `run18 — V5_fullstart_c64b6_cold_nogate_fixed`.
- `07-09` **Verification battery: 3/3 PASSED** (same frozen net pair as the broken-era tests — pure
  driver A/B): (1) same-net v800-vs-v1 **3.5% → 98.0%** (+98 −2, Elo +676; W 50-0, B 48-2);
  (2) net59-vs-net1 @128v **0.6% → 97.8%** (+391 −9, Elo +655; as-Black 200-0 sweep) — the pt/bin
  "harness split" is dead, C++ now agrees with the python anchors; (3) same-net null **51.6%**
  (+206 −193 =1, Elo +11, W-by-color ~77% both slots — symmetry preserved). Instrument certified;
  `cc strength` rows are trustworthy from cand ≥ #90.

## Result

<active — leave empty. Primary read: post-fix cumElo slope of the promote-always series + anchor
trajectory (`anchor_gauntlet.jsonl`) vs the run-17 curve. Link successor when pivoted.>
