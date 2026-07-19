# Run 22 — distinguishable candidates + high-power gate (q-blend carried)

> Run 21's stack (panel + PFSP league + `VALUE_Q_RATIO=0.5`) + the two
> candidate-distinguishability fixes from the 07-14 diagnosis day: **EMA decay
> 0.999 → 0.99** and **publish every 200 games** (candidates stop being ~88%
> identical twins), plus a **4× gate** (160-game main match, 40-game panel legs)
> so the gate measures real deltas instead of coin-flipping between near-clones.
> Motivating evidence in run19.md (LR-drop probe + chunk forensics) and run21.md.

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` (DB) | `run22_V5_fullstart_c64b6_qgate` (training-run id 1, dir `run1`) |
| Start FEN | official full start (`STARTING_FEN`, `{wm:2}`) (= runs 19–21) |
| Arch | SE-ResNet gather head, c64/b6 ~630K, tag `v5` (= runs 18–21) |
| Optimizer | Adam, lr=1e-3 (LR-drop rule **retired** — see Decision rules) |
| Policy / value targets | `visits` / **z↔q blend `VALUE_Q_RATIO=0.5`** (carried from run 21's resume; q plumbing sign-audited 07-14) |
| Init | COLD random init (= runs 18–21 pairing) |
| **Gate** | −20 lenient gate, **`matches.games` 8 → 32 (×5 slice-0 ⇒ 160 games, σ≈27 Elo — was 40g/σ≈55)** + regression panel **`panel.games` 4 → 8 (⇒ 40 games/leg × 2, σ≈55/leg, thr −50 ≈ 0.9σ)** |
| **Publish / EMA** | **`PUBLISH_GAMES` 100 → 200** (≈256 trainer steps/candidate) + **`EMA_DECAY` 0.999 → 0.99** (t½≈69 steps ⇒ consecutive candidates ~8% parent carryover, was ~88%) |
| League | enabled, fraction 0.2, poolSize 8, `pfsp: true` (= run 21) |
| Rules | v6 bottom-*d* charge (= runs 18–21) |
| Replay buffer | unchanged (window ramp 400→4000 @α0.75, RF=8, batch 1024) |
| Key trees | fork `709d060` + client `1d56ccd` (= run 21); server = `9168d59` + serverconfig gate bump + `restart_fleet.sh` EMA/PUBLISH forwarding (uncommitted); engine = `b0bdef7` + `cc.py --ema-decay/--publish-games` flags (uncommitted) |
| Fleet box | vast `44287736` (RTX 3060 — same box as runs 18–21) |
| Started | 2026-07-15 |
| Status | **concluded 2026-07-19** (user pivot) → run 23 |

## Hypothesis

Runs 19–21 gated **near-clone candidates with an underpowered instrument**: EMA 0.999 at
fleet step rates (~130 steps/publish vs 693-step half-life) made consecutive candidates
~88% the same weights, and 40-game matches (σ≈55) turned the −20 gate into a coin flip
(stall-floor promotions at E[+32]/promotion). Run 22 makes candidates ~92% fresh and
more than halves gate noise, so gate verdicts should finally track real strength deltas.
The q-blend value target (first real runtime here — run 21 was too short) should keep
the value head learning where pure z degenerated into outcome-noise memorization
(LR-probe verdict, run19.md).

Success reads: (a) gate promote/reject decisions **correlate with the anchor slope**
(promotions cluster while seed13 climbs, rejections while flat) instead of a constant
~64-80% promote rate; (b) daily `cc champs` audits show an **ordered field** (best
actually measures first; spread > multiple-comparison noise) vs run 19's 81-Elo
scramble with best last; (c) cum gate Elo de-inflates toward anchor-measured truth
(stall-floor inflation drops from ~+32 to ~+11/promotion at σ≈27). Failure reads:
promote rate collapses while anchors climb (gate now too strict → revisit thresholds,
not games); or candidates distinguishable but champs audit still scrambled ⇒ the
residual noise was never the gate — data-side story only.

Costs accepted: gate load ≈ 240 match games/candidate @128v ≈ ~40 selfplay-game
equivalents per 200 training games (~17% throughput) plus match-priority pauses;
fewer, chunkier promotions (league pool grows slower early).

## Design delta vs run 21

- `serverconfig.json`: `matches.games` 8→32, `matches.panel.games` 4→8 (thresholds
  unchanged: −20 main / −50 panel).
- `launch_trainer.sh` env via fresh-run: `EMA_DECAY=0.99`, `PUBLISH_GAMES=200`
  (launch_trainer defaults still 0.999/100 — the values ride the knob env).
- `restart_fleet.sh`: now **forwards `EMA_DECAY` + `PUBLISH_GAMES`** into the trainer
  pane (they'd previously have silently reverted to defaults on a reboot), with
  run-22 defaults baked in per the can't-silently-revert rationale.
- `cc.py fresh-run`: new `--ema-decay=` / `--publish-games=` flags → knob env → launch
  line + @reboot cron.
- `VALUE_Q_RATIO=0.5` carried from run 21's resume (was flipped mid-run-21; here from t=0).
- Nothing else: same arch, optimizer, buffer, start FEN, league/PFSP, rules, box.

## Log

- `07-15` Run 21 concluded (~00:25; too short to read PFSP/q-blend — served as the
  07-14 diagnosis-day vehicle) and archived to
  `~/chessckers-backups/run21-fullstart-c64b6-pfsp-20260715/`; fleet clean-stopped,
  snapshot verified complete before the wipe.
- `07-15` `cc fresh-run --run-name=run22_V5_fullstart_c64b6_qgate --arch=v5
  --c-filters=64 --n-blocks=6 --se-ratio=8 --value-q-ratio=0.5 --ema-decay=0.99
  --publish-games=200 --parallelism=32` (cold). Provision rsync carries the
  serverconfig gate bump + restart_fleet forwarding to the box.
- `07-15` First reject: #22 passed the main gate vs #21 (95-65, +66) but panel leg
  vs champ #20 came in 16-24 (−70 ≤ −50) → reject; second leg (vs #19) auto-skipped.
  Textbook RPS triangle (#22 > #21, #21 > #20 +39, #20 > #22 +70) — the panel doing
  its run-20 job. Tooling: `cc strength` now prints panel legs dim+indented under
  each gate match (W-L-D, calcElo, ok/FAIL vs `matches.panel.threshold`, skipped
  legs marked), so panel rejects read directly off the table.
- `07-16` **Champs audit sharpened: 12 → 40 games/pair** (`champ_ladder.py --games`
  default; the 04:45 cron now inherits it). Motivation: the 07-16 12g audit was
  unreadable — 8 nets inside 84 Elo with best ranked 6/8, but per-net 95% CI ≈ ±80
  at 12g, so ordered-field/RPS reads (decision rules below) can't fire either way.
  At 40g: ≈ ±40 Elo/net; nightly cost 1120–1440 games ≈ 2.8–3.6h at the measured
  ~9s/game (clears the 08:15 anchor cron). Game draws are a non-factor (11/408 =
  2.7% across the first two audits). Don't trend `spread`/`best_rank` in
  `champs_audit.jsonl` across the 07-16 boundary — the error bars halve there.
  **[Superseded same-day by the harness finding below: ALL engine-ladder Elo —
  any games count — was measuring a broken operating point.]**
- `07-16` **"#27 beats everything after it" investigated → cc champs harness
  invalidated; two root causes found, one fixed.** Operator observed champ #27
  consistently topping `cc champs` fields. Tournament-mode rematch (the gate's own
  harness — client launches `selfplay --player1/--player2 --no-share-trees`,
  matchParams temps) says the opposite: **#27 vs #52 = 26.4% over 159g @128v
  (−178, LOS 0.00) and 15.8% over 38g @800v (−291)** — gap concentrated as Black
  (#27-as-Black 6% @128v, 0% @800v). De-inflated gate ratchet (~25 promotions ÷2.7)
  predicts ≈ +200, matching: **the gate has made real progress in its own harness;
  league (800v, learner-POV) concurs (later nets ~65% vs #27, n=68).** The same
  pair through the UCI ladder @128v: **53%/+10 for #27** — a ~190-Elo harness split
  on identical .bins. Root cause 1 (FIXED): PyVariant FENs froze `fullmove` at 1
  (Black moves bypass `board.push`), and the fork derives `GetGamePly` from that
  field under stateless `position fen` driving → **temperature never decayed —
  every historical `cc champs`/`cc ladder --engine` game ran full-noise end-to-end**
  (Black-blunderfeast: 94% Black wins vs the fleet's ~70% White; skill gaps
  compressed → scrambled audits, old-above-new mirages, run 19's "flat field"
  verdict included). Fix: `moves_black.py` ticks `fullmove_number` on Black move
  completion (tests pass; deployed to box). Root cause 2 (REMAINS, structural):
  post-fix rematch @128v flips harder — **White won 0/71** (n27-as-W 0-19-17,
  n27-as-B 35-0-0; draws = 400-ply cap; games 9s→51s) vs tournament's 70% White.
  Fresh-tree-per-move UCI (`position fen`, no reuse) is a different operating
  point from tournament tree-reuse search: White's conversion plan needs the
  compounded depth, Black's mandate chains don't (consistent with 800v amplifying
  the strong side). **Consequences:** (a) treat all historical engine-ladder Elo
  (champs audits, run-19 champ-ladder conclusions) as void; (b) the audit must run
  in the gate's harness — port `champ_ladder` match play to fork selfplay mode
  (TODO), or teach `engine_uci` history-based `position ... moves` driving;
  (c) the plateau narrative REOPENS: seed13-flat is a python fresh-tree instrument
  too — production-harness evidence (gate, league, this diagnostic) shows steady
  real progress (~+7/promotion); (d) `play_tui`/stateless-UCI opponents also
  benefited from the fullmove fix. Diagnostic artifacts on box:
  `/workspace/diag27_sp{128,800}.log`, `/workspace/diag27_uci128{,_fixed}.log`.
  **Fallout fix (same day):** the fullmove tick broke `cc games` chunk-move
  recovery and would have false-flagged `cc verify-chunks` — the fork's chunk
  FENs freeze fullmove at 1, and the transition-matcher compared clocks strictly
  (worked pre-fix only because both writers were frozen). `check_chunk_parity._norm`
  (shared canonicalizer) now blanks halfmove/fullmove alongside the dead ep field;
  board/overlay/turn/castling stay strict. Verified on training.10999.gz: 61/61
  moves recovered, parity clean; tests pass; synced to box. A stale-checker run
  in the fix window flagged "tons" of ILLEGAL plies (all Black plies — the
  fullmove signature, incl. exotic-but-legal sprint/deploy-on-king/hop-with-
  promotion transitions in training.10827.gz); re-verified with the fixed
  canonicalizer: 10827 clean + **60-newest-chunk sweep clean** — fork↔PyVariant
  parity is intact, training data unaffected.

- `07-17` **Plateau probe: #75 vs #52, tournament mode @128v (diag27 recipe, 160g)
  — #75 wins 99-60 (62.3%, +87 Elo, LOS 99.9%).** The climb continued through the
  newest 23 candidates (~+3.8/candidate — decelerating vs ~+7.1/candidate for
  #27→#52, but unambiguously positive). Seed13-flat + champs-audit "plateau" reads
  stay instrument-side (the 07-16 40g audit had already self-flagged via
  `white_share_alert`: 0.239 vs fleet 0.751). Color split — #75 better on BOTH
  colors: as White 76-4 (95%) vs #52-as-White's 56-23 (71%); as Black 23-56 (29%)
  vs #52-as-Black's 4-76 (5%). White won 83% overall, matching fleet balance (84%),
  so the conversion-monoculture ceiling story remains the live watch item. Probe ran
  in tmux `diag75`, fleet undisturbed. Artifacts: `/workspace/diag75v52_sp128.log`.
- `07-17` **`cc champs`/`cc ladder --engine` ported to the gate harness** (the 07-16
  TODO): `ladder.py` engine mode now plays each pair via the fork's own
  `selfplay --player1/--player2 --no-share-trees` tournament (matchParams temps,
  per-pair `--parallelism` 32, scored from the engine's `tournamentstatus` stream —
  NB the engine omits the `Elo:` field at 100%/0% scores, so the parser keys on the
  W/L/D triplets; line shapes pinned in `tests/test_ladder_tournamentstatus.py`).
  Per-net `@VISITS` maps to `--playerN.visits` (verified supported). Stateless UCI
  driving survives as `--harness uci`, diagnostics only. Box smoke (c73/r74/best ×
  8g): matrix/Elo/jsonl all render, **White share 79% vs fleet 76% → calibration
  tripwire PASSES** (the invalidated harness ran 24%). The 04:45 cron now measures
  the promotion operating point — first trustworthy `champs_audit.jsonl` row lands
  tonight; don't trend spread/best_rank across the 07-17 boundary (earlier rows
  void per the 07-16 finding).
- `07-17` **`cc play` ported to the fork** — the TUI's opponent had been the
  *Python PUCT reference* (200 sims, CPU) all along, not the production engine.
  `play_net.py` now defaults to the fork (auto-detects the sibling build, Mac or
  box; `.pt` auto-exports to `.bin`) driven with FULL HISTORY —
  `engine_uci.bestmove(fen, moves=[...])` sends `position fen <start> moves ...`,
  so the fork sees the true game ply and reuses its tree between its moves
  (interactive play can't use tournament mode; history driving is the
  production-like operating point for it). Crash-lossless: full-history resend
  after `restart()` replays the game; undo = truncated history (verified).
  Validated on the Mac Metal build: 12-ply self-drive all PyVariant-legal (incl.
  the wm:2 double move), piped-TUI end-to-end, undo, `--mcts` fallback intact.
  **The 800v UCI hard-crash reproduces in history mode too** (silent death, ply 3)
  — default stays `--visits 128`; the crash is a standing fork bug. Python MCTS
  opponent survives as `--mcts` (it renders the WDL/top-lines panels; fork mode
  is board-only).

- `07-18` **The "tree reuse segfault" root-caused: TWO stacked >255-legal-move
  bugs in the fork's classic search, both fixed** (both live in every run-16+
  binary, incl. run 22 production until deploy):
  1. The June uint8 `num_edges_` overflow fix (0eaca61) was silently LOST in the
     run-16 branch rewind (dangling commit, reachable from no branch — same
     failure mode as the provision-script reverts). Every >255-legal-move
     position truncated its move list mod 256: silent policy/move-set corruption
     in training data, and count%256==0 → edge-iterator wrap → wild-pointer
     segfault. Restored by cherry-pick → fork `f8e19d1`.
  2. Upstream lc0 hardcodes 256-entry per-node scratch arrays in
     `PickNodesToExtendTask` (chess max 218 legal moves; Chessckers exceeds it):
     once num_edges_ is uint16, any node whose `max_needed = NStarted +
     cur_limit + 2` crosses 256 writes OOB (stack + heap) → bad_alloc/SIGSEGV.
     This is WHY crashes tracked "tree reuse @ 800v": NStarted accumulates on
     reused/shared trees; 128v fresh-tree gate/ladder games never cross 256.
     Fixed: `kMaxNumEdges=1024` sizes all per-edge scratch (classic +
     dag_classic incl. CurrentPath index_ 8→10 bits) + `CreateEdges` clamps
     wider lists with a CERR warning → fork `fec5291`.
  Deterministic repro (PyVariant-constructed, 300 legal moves, fork parity
  exact post-fix):
  `K7/4k3/1k6/5k2/3k4/1k6/6k1/8[b3:kkkkk,d4:kkkkk,f5:kkkkk,b6:kkkkk,g2:kkkkk,e7:kkkkk] b - - 0 1`
  — pre-fix binary saw 44/300 root moves; post-uint16-only it corrupted the
  heap (bad_alloc after first 800v search). Mac Metal verify battery: 300/300
  root moves, 3× stateless 800v, 16-ply history-driven game, shared-trees 800v
  tournament soak — all clean. Deploy via `./run.sh` (rebuild + CUDA probes
  incl. the repro FEN + client-only relaunch). **Data caveat: run-22 games
  before the deploy carry truncated move lists at >255-move positions** (rare;
  late tall-King-tower middlegames), and prior crash telemetry blamed on GPU
  pressure likely includes these. `--no-share-trees` in ladder/champs remains
  only for search independence, no longer crash avoidance.

- `07-18` **League-selfplay throughput benchmark (quiesced box) → production config
  confirmed optimal.** Method: fleet + audits stopped, 7 time-boxed configs of the
  fork's `--training=true` selfplay with the live client's exact flags (league 0.2,
  7-net PFSP pool); throughput = tail-window slope of `gameready` completions
  (`engine/scripts/bench_selfplay.py`; raw logs + results.jsonl archived at
  `engine/telemetry/bench-league-20260718/`). At 800v: P16 4.37 / **P32 4.97** /
  P48 4.70 g/min — parallelism 32 is the peak and the curve is flat. League off:
  4.84 (league sampling ≈ FREE, within noise). `--no-share-trees`: 3.94 (**tree
  reuse = +26% throughput** — the 07-18 crash fixes protect real speed, not just
  stability). Minibatch 128: 4.76 at 2× VRAM (no). 128v smoke: 19.7 (4.0×, not
  6.25× — reuse amortizes deep searches). GPU util pinned 76–83% in EVERY config →
  serving-path/CPU-bound, not GPU-bound; that ceiling is where any future win lives.
  Environment cost: isolated 298 g/hr vs ~164 g/hr observed in-fleet → trainer +
  gate + audit contention was eating ~45% of selfplay throughput. Worst offender:
  the 04:45 champs audit still running at 15:24 (10h39m, pairs hitting the 85-min
  watchdog under contention) plus 9 idle UCI-mode engines pinning ~5GB VRAM — spawned
  by the pre-05:41 champ_ladder it held in memory (eager per-net UciEngine
  construction; absent from the current selfplay-harness code path, re-verified
  post-restore: 0 UCI procs during a live champs run). Fleet restored 17:23 with
  the run-22 env verbatim (q-ratio 0.5 trap avoided); queued #83/#84 gates played
  first.
- `07-18` **Publish cadence 200→400 games** (the candidate-distinguishability rule's
  pre-committed remedy, triggered by the throughput read): at 200, each candidate
  cost 240 gate+panel games @128v ≈ 12 min GPU per 40 min of training games (~23%
  of GPU time), while recent candidates landed −13…+30 vs best — inside the gate's
  ±28 Elo 1σ @160g, i.e. noise-dominated promotions (#82 promoted while losing
  77-83). At 400: gate share ~13% (≈ +13% training throughput) and per-candidate
  deltas double → signal-dominated gate, EMA twins separate further. Cost accepted:
  best-net feedback latency ~81 min → ~2.5 h. Set in the restore env, the `@reboot`
  cron line, and `restart_fleet.sh`'s default (all three track the run).
  **Warm-restart quirk observed at the 17:23 restore:** a trainer restart costs TWO
  spurious candidates — the documented initial publish (#84, 17:23, promoted → best)
  plus one more (~27 min later, #85) when the bridge re-feeds the chunk backlog and
  trips the fresh games-since-publish counter (buffer snapshot off in this run →
  counters don't resume). ≈ 480 gate games per restart; budget for it, don't
  diagnose it as publish-cadence breakage — the 400-game counter is clean from the
  second artifact onward.
- `07-18` **Anchor-gauntlet cron retired; absolute rung moved into the champs audit
  (`--pin`).** The 8-hourly pure-python gauntlet (PyVariant MCTS + CPU alpha-beta)
  was the box's last cron-driven python game-player and a contention source; removed
  from crontab (nothing reinstalls it; `cc anchor` stays for on-demand absolute
  reads). Replacement: `champ_ladder.py --pin N` freezes net #N under
  `networks/pins/` + run-scoped pins.json, auto-included in every later audit of the
  run. Run 22 pins **p2** (the first promoted champion) → best-vs-p2 in
  champs_audit.jsonl is the run's absolute trajectory. p2 is already saturated
  (best 8/8 in the smoke check) — it is a REGRESSION TRIPWIRE and continuity rung,
  not a discriminator; add a mid-strength pin (`--pin 52`-era) if discrimination is
  wanted. AMENDS the plateau/anchor rules below: "anchor rows" now come only from
  on-demand `cc anchor` runs; day-to-day plateau reads = champs-audit
  spread/ordering + the p2 regression watch.
- `07-18` **Anchor gauntlet grew per-color + truncation + start-FEN fields; the first
  instrumented row shows the gauntlet sits in the fresh-tree White-collapse regime.**
  `anchor_gauntlet.py` rows/prints now carry the current net's W-D-L split by color
  (`as_white`/`as_black`), per-anchor ply-cap truncation counts (`trunc`), and
  row-level `start_fen`+`max_plies` (previously unrecorded operating-point
  variables). First row (net #88, 34g vs seed13 @100 sims): **as-White 0-4-13,
  as-Black 12-5-0 — zero White-side wins in either direction**; all 9 draws are
  160-ply caps; the search:3 control (as-W 9-1-0) shows the harness isn't
  Black-biased per se, only net-vs-net is. This is the 07-16 root-cause-2 signature
  (UCI fresh-tree @128v: White 0/71 vs tournament's 70% White) extended to the
  python gauntlet, which is fresh-tree by construction (`NetPlayer` → `pick_puct`
  per move, no reuse): White's conversion needs compounded depth, Black's mandate
  chains don't. Consequence: the seed13 aggregate is two near-symmetric Black-side
  scores canceling (run22-as-B 85% vs seed13-as-B 88% here) — the rung's
  net-strength signal at this operating point is ~nil, and the historical flat
  −20…−95 band reads as the op-point signature, not (only) a plateau — hardening
  (c) in the 07-16 bullet. The fleet's "82% White conversion" is a
  tree-reuse/800v-regime fact; cross-net at shallow fresh-tree it inverts. Next
  reads: on-demand 800-sim fresh-tree probe (separates visit count from tree reuse
  as the collapse driver — 07-16 only measured fresh-tree @128v), and/or a seed13
  rung inside the champs (gate-harness) audit, which needs champ_ladder support for
  external-net pins.
- `07-19` **800-sim fresh-tree probe: visits exonerated — tree reuse is the
  load-bearing ingredient in White's conversion.** On-demand `cc anchor --anchors
  seed13 --sims 800 --games 16` (fresh-tree python MCTS, net #91, own history file
  `anchor_probe_800.jsonl`): **as-White 0-2-6, as-Black 5-3-0 — still zero
  White-side wins in either direction** (11/16 decisive, all to the Black side;
  5 draws = 160-ply caps). With the fork's post-fix data this completes the grid:
  reuse @128v → 70% White (tournament), reuse @800v → 82% White (fleet selfplay),
  fresh @128v → White 0/71 (fork UCI), fresh @100 sims and fresh @800 sims → zero
  White wins (python gauntlet). White's born +0.5 is only cashable with search
  compounded across moves; no practical per-move visit count rescues a fresh tree.
  Consequence: the python gauntlet cannot be made strength-informative via
  `--sims` — Black-side scores stay near-symmetric (run22-as-B 81% vs seed13-as-B
  88%, n=8 each) and the aggregate (−22 [−186, +143]) reads op-point, not
  strength. Absolute rungs belong in the gate harness: champ_ladder needs
  external-net pin support (freeze seed13's .bin into the audit); the python
  gauntlet stays a rules-level sanity instrument only.
- **LR drop — RETIRED.** The 07-14 probe (run19.md) refuted "drop LR at plateau":
  3e-4 on the frozen plateau window cut value loss 63% and played 45.3% (−33±20) vs
  its own start. A plateau is **data-side until proven otherwise**. On plateau
  (both-instruments definition below): do NOT touch LR; the next lever is
  data/target-side — balanced or Black-favorable opening seeding (design sketch in
  run19.md chunk-forensics bullet) — or conclude the run.
- **Plateau definition** — seed13 flat (95% CI overlapping zero gain) ≥3 anchor rows
  AND daily champs audit spanning <100 Elo across the stretch. Floor guard applies
  (score ≤0.05 rows are unmeasurable, not flat).
- **Gate health (new, replaces run 20's panel-only rule)** — expected promote rate for
  a true-equal candidate at σ≈27/thr −20 is ~77%; panel false-rejection ~18%/leg on
  true-equals. ≥5 consecutive rejections while anchors CLIMB = gate too strict →
  loosen `panel.threshold` toward −20 before touching games. Rejections while anchors
  are flat = working as designed (run 20's lesson).
- **Candidate-distinguishability check (new)** — after ~10 promotions, `cc champs` on
  consecutive candidates: if the field still spans < the gate's σ, EMA/publish did not
  separate candidates → revisit (EMA 0.99→0.98 or publish 200→400) with a dated bullet.
- **RPS check cadence** — daily champs audit; cycles beyond noise over ≥3 audits →
  PFSP probs investigation (run 21's rule verbatim).
- **PFSP probs sanity** — run 21's rule verbatim (`[league] pfsp probs` mass on
  hardest, ε floor visible, investigate result plumbing if ~uniform after ≥200
  result-bearing league games).
- **Anchor rotation / abandon** — run 20's rules verbatim (auto-pin at saturation;
  48h-clearly-slower-than-run-19-at-matched-nets with gate exonerated → pivot).

## Result

**Concluded 2026-07-19 by user decision (pivot to run 23) — the gate-and-instrument
overhaul run; the q-blend value read left incomplete.** ~4.5 days, best ≥ #91
(final endpoint counts: stamp from the archived DB). What it delivered:

1. **Gate machinery validated in production** — first-ever panel rejection on a live
   RPS triangle (#22, 07-15); publish 200→400 mid-run moved promotions from
   noise-dominated to signal-dominated (gate ≈13% of GPU); the warm-restart
   two-spurious-candidates artifact documented.
2. **Instrument stack rebuilt** — champs 12→40g/pair, then the harness invalidation
   (PyVariant fullmove freeze → all historical engine-ladder Elo VOID; fixed),
   champs/ladder ported to the gate harness, anchor cron retired → champs pins,
   per-color + trunc + start-FEN fields added to the anchor gauntlet.
3. **The fresh-tree White-collapse discovery** (07-18/19) — cross-net fresh-tree play
   at the full start yields ZERO White-side wins at 100 AND 800 sims; tree reuse
   (not visit count) is what cashes White's born +0.5; the fleet's 82% White
   conversion is regime-specific; the python gauntlet retired as a strength
   instrument (rules-sanity only) and the run-19-era "seed13 plateau" reads
   reclassified as op-point signatures.
4. **Run-15's Gumbel verdict audited** — its match-based evidence (freeze at #5,
   −98→−301, as-Black collapse) is blacks_move-era contaminated (re-test candidate
   per the run-14 postmortem; retro-caveat added to run15.md); the chunk-level
   c_scale=1 one-hot arithmetic stands. This audit motivated the pivot.
5. **q0.5 blended value target** — ran the full run without incident but the formal
   `value_sign_agree` read was never taken; carried as an open question for the next
   full-start run (run 23 uses pure-z per the Stage-1 design).

Successor: **run 23** (`run23.md`) — Gumbel Stage-1 re-test @ c_scale=0.1, cold, on
the e8/d8 KK-vs-K start (known ground truth, fast games). Archive:
`~/chessckers-backups/run22-fullstart-c64b6-qgate-20260719/` (DB, trainer/run1,
networks, games, telemetry).