# Run 21 — league PFSP (win-rate-weighted opponent sampling)

> Run 20 (gate regression panel) + exactly one training-loop change: **PFSP league
> weighting** — league opponents are sampled by live per-opponent win rate
> (`f_hard(wr)=(1−wr)²` on Laplace-smoothed 48h scores + 20% uniform floor) instead of
> uniformly. Run 20's 10 hours produced the cleanest RPS evidence yet: candidates
> #10/#11/#13/#14 drew-or-beat best #9 while losing −53…−191 to panel champ #8 — the
> exact "old champion the line forgot how to beat" matchup PFSP concentrates league
> games on. Design in `league-selfplay.md` (PFSP section).

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` (DB) | `run21_V5_fullstart_c64b6_pfsp` (training-run id 1, dir `run1`) |
| Start FEN | official full start (`STARTING_FEN`, `{wm:2}`) (= runs 19/20) |
| Arch | SE-ResNet gather head, c64/b6 ~630K, tag `v5` (= runs 18–20) |
| Optimizer | Adam, lr=1e-3 (= run 20; LR-drop rule inherited below) |
| Policy / value targets | `visits` / pure z (`VALUE_Q_RATIO=0`) (= runs 19/20) |
| Init | COLD random init (= runs 18–20 pairing) |
| Gate | −20 lenient gate + regression panel (= run 20: `panel {enabled, 2, 4×5=20, −50}`) |
| **League** | enabled, fraction 0.2, poolSize 8 + **`pfsp: true`** — probs from last-48h league win rates, `f_hard` p=2, ε=0.2 floor, cached per (best, pool), shipped `/next_game leagueProbs` → client `--league-probs` → engine inverse-CDF |
| Rules | v6 bottom-*d* charge (= runs 18–20) |
| Replay buffer | unchanged (window ramp 400→4000 @α0.75, RF=8) |
| Key trees | engine `b0bdef7`, server `9168d59`, fork `709d060`, client `1d56ccd` |
| Fleet box | vast `44287736` (RTX 3060 — same box as runs 18–20) |
| Started | 2026-07-14 |
| Status | active |

## Hypothesis

Uniform league sampling spends most league games on opponents the learner already
crushes (run 20: 229 league games spread 12–51 per opponent regardless of strength).
PFSP concentrates them on the measured-hardest pool members, so the training data
itself contains the matchups the panel keeps rejecting candidates over. Success
reads: (a) run-20-style panel rejection streaks resolve faster — after a "passed
best, regressed vs champ X" rejection, league sampling visibly shifts onto X
(`[league] pfsp probs` log) and a later candidate clears X; (b) anchor trajectory ≥
run 19's archived curve at matched net count; (c) champs-audit cycles
(A>B>C>A) rarer than run 19's. Failure reads: probs pile onto one opponent with no
panel-clearance improvement (floor too low / window too long), or no behavioral
difference vs run 20's uniform league at matched net count.

## Design delta vs run 20

- **League PFSP** (engine `709d060`, client `1d56ccd`, server `9168d59`; design
  `league-selfplay.md`): client forwards gameready `result`/`player1` tokens →
  `training_games.result`/`learner_is_black` (AutoMigrate) → server computes
  per-opponent probs, cached per (best, pool) so `/next_game` stays byte-stable
  between promotions → `--league-probs` → engine weighted sampling (validated,
  fatal on count mismatch; empty = uniform).
- Nothing else: same arch, optimizer, targets, buffer, start FEN, gate+panel,
  league fraction/pool, box as run 20.

## Log

- `07-14` Run 20 concluded (~10h; panel validated — 4 stall-floor coin-flips
  rejected; RPS cycle #9-vs-#8 documented) and archived to
  `~/chessckers-backups/run20-fullstart-c64b6-panel-20260714/`; fleet clean-stopped.
- `07-14` `cc fresh-run --run-name=run21_V5_fullstart_c64b6_pfsp --arch=v5
  --c-filters=64 --n-blocks=6 --se-ratio=8 --value-q-ratio=0 --parallelism=32`
  (cold). Provision rsyncs carry PFSP code + `serverconfig.json pfsp: true` → PFSP
  live from t=0 (dormant until the pool exists, ~2nd–3rd promotion; probs ~uniform
  until result-bearing league games accrue — by construction, never garbage).

## Decision rules (pre-committed — inherited from run 20 + PFSP-specific)

- **LR drop / plateau / anchor rotation / abandon** — run 20's rules carry over
  verbatim (seed13 slope trigger with floor guard, both-instruments plateau
  definition, auto-pin logging, 48h-slower-than-run-19 pivot).
- **Panel health** — run 20's rule, with the run-20 lesson attached: a rejection
  streak only indicts the panel if anchors are *climbing* through it.
- **PFSP probs sanity** — after each promotion, the server logs `[league] pfsp
  probs …`. Expect mass on the measured-hardest opponent(s), every opponent ≥
  ~0.025 (ε floor), and ~uniform right after pool birth. If probs stay ~uniform
  after ≥200 result-bearing league games (`select count(*) from training_games
  where opponent_network_id>0 and result>0`), investigate the result plumbing
  before blaming the math. If probs pin one opponent >0.8 for >24h with no panel
  improvement, consider ε 0.2→0.3 or window 48h→24h; log any change here.
- **RPS check cadence** — daily champs audit jsonl (cron: operator step below);
  run-20's #9/#8-style cycle recurring *despite* PFSP over ≥3 audits = escalate to
  matrix cross-table / Nash-pick design (deferred idea).

## Result

<active — primary reads: anchor trajectory vs run 19/20 archived jsonls (`cc
compare`), panel-rejection-streak length vs run 20, `[league] pfsp probs` shift
after each panel rejection, champs-audit spread/cycles.>
