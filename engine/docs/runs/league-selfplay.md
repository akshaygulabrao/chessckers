# League self-play — population-based opponent sampling (anti-RPS)

> **LANDED 2026-07-11** — committed `league.enabled=true`, so it goes live automatically
> when **run 19** provisions (the run-18 box's config predates the block and is untouched).
> Implemented across all three fleet repos on the run-18 lineage branches
> (engine/server: `ctl/pre-gumbel-run16`, client: `chessckers-port`). Verified end-to-end
> locally (standalone engine smoke + local mini-fleet). Run 18 was left untouched as a
> clean experiment.

## Why

Pure self-play + a candidate-vs-single-best gate cannot distinguish "stronger" from
"a better counter to the one opponent it trained against." Candidates are trained on data
generated around the current best's play distribution, then gated against exactly that net —
the maximally rock-paper-scissors-exploitable loop. Promote enough narrow counters and the
chained gate-Elo climbs while absolute strength cycles (the `cc anchor` instrument exists
because of this: chain-Elo and fixed-anchor Elo told opposite stories in run 17).

League self-play attacks the **generator**: a fraction of training games are played
current-best vs a **past champion** sampled per game from a pool, so counter-one-opponent
strategies stop being rewarded in the data itself (OpenAI Five ran 80/20 self/past;
AlphaStar's league is the maximal version). Decisions taken: **both sides' plies** enter
training data (no learner-side filtering — explicit user call), and the gate itself is
unchanged (a regression panel remains a separate TODO in `lczero-server/main.go` ~L485).

## Mechanics

Per-game opponent sampling lives **inside the engine's selfplay tournament** — with a 1-client
fleet and a long-lived engine process (parallelism 16-32), server-side per-assignment rotation
would give coarse mixing and kill in-flight games on every rotation. (Upstream lc0's dormant
`MultiNetMode` in `nextGame` does per-*client-token* opponent pinning — it mixed across
thousands of clients and degenerates on ours; left untouched/off.)

- **Server** (`lczero-server`): `leaguePoolShas()` in `main.go` reconstructs past champions
  from passed gate matches and picks **log-spaced** ones (1, 2, 4, 8, … promotions back,
  newest first, current best excluded, capped at `poolSize`). Training `/next_game` responses
  gain `leaguePool` (shas) + `leagueFraction` when enabled and ≥2 promotions exist. The pool
  depends only on the matches table + best id, so responses are **stable between promotions**
  (the client restarts the engine whenever the response changes — a jittery pool would restart
  it every 60s poll). `upload_game` accepts `opponent_sha` → new indexed column
  `training_games.opponent_network_id` (0 = self-play; gorm AutoMigrate adds it on restart).
- **Client** (`lczero-client`): downloads pool nets (cached, `inf` keepTime), passes
  `--league-weights=<paths>` + `--league-fraction=<f>`, extends the 60s-poll restart check to
  the league fields, parses the gameready `opponent <k>` token, uploads `opponent_sha`.
- **Engine** (`akshay-chessckers-0`): `--league-weights` / `--league-fraction` selfplay
  options; pool backends loaded once (shared_ptrs); per-game coin flip in `PlayOneGame`
  swaps **player2's** backend to a uniformly sampled pool net — player1 (the learner) keeps
  the current net and the existing color alternation, so the learner alternates white/black
  across league games. League games force **separate search trees** (`kShareTree &&
  league_idx < 0`) — different-net players must not share a tree (known segfault class).
  League gameready lines carry ` opponent <k>` between the `gameid` value and
  `play_start_ply` — the one slot the client's index-based parser never slices.
- **Trainer**: untouched. League chunks are byte-identical ccz1; the bridge feeds by
  sequence number, lineage-agnostic.

## Config (`lczero-server/serverconfig.json`)

```json
"league": { "enabled": true, "fraction": 0.2, "poolSize": 8 }
```

Enabled from t=0 of a fresh run is safe: the pool is empty until the 2nd promotion, so games
are pure self-play and league phases in by itself. To toggle mid-run: edit the on-box
serverconfig + restart the server pane (state preserved); clients pick it up within 60s and
cleanly restart engines.
Note the **bootstrap champion** (first net, promoted matchless) never enters the pool —
champions are reconstructed from passed matches only.

## ⚠ Deploy order

**Never run a new client against an old engine binary** — the client passes
`--league-weights`, an old engine prints "Unknown command line flag", and the client's
stdout scanner `log.Fatal`s. Deploy engine before (or with) client. Old-client/new-server
and new-client/old-server are both safe (unknown JSON fields ignored / fields absent).
Also: two-net league games require the `ee64b19` blacks_move fix (already on the run-18
lineage) — without it, `{wm:2}` double-moves route plies to the wrong net's search.

## Monitoring

- `fleet_status.py` (via `cc status`): `league: N games vs past champions | last 1h n/total`.
- Attribution: `sqlite3 chessckers.db "select opponent_network_id, count(*) from
  training_games group by 1"` — per-opponent counts (feeds a future PFSP sampler).
- Engine relaunch check: `grep league-weights client.log`.
- **Success criteria for the feature** (run 19+): `cc anchor` trajectory (fixed anchors)
  keeps rising rather than flattening while gate-Elo climbs; `cc ladder` round-robins show
  fewer A>B>C>A cycles among successive champions.

## Verification record (2026-07-11, this Mac)

Standalone engine: 24 games at `--league-fraction=0.5` → 12 league games, token placement
exact, non-league lines byte-identical, chunks decode with every ply of both sides
(1448 examples = 1448 moves), no crashes on the separate-trees path; fraction=0 and
no-flags controls clean. Mini-fleet (server+client+engine, sqlite, promote-always,
~8 min): 3 nets seeded → 2 gate matches promoted B then C; pool advertised exactly
{B} (current best correctly excluded); engine relaunched with `--league-weights` 26s
after promotion; **52/111 training games (46.8%) league at fraction 0.5**, every one
attributed to B's id (`opponent_network_id` ∈ {0, B} only); `fleet_status` league line
rendered; repeated `/next_game` responses byte-identical; 0 server/client/engine errors.
Notes: ccz1 chunks carry no top-level `outcome` key (result is baked into per-example
`wdl_target`); a mixed-arch gate opponent (old run-6 net) loaded and played fine —
.bin is self-describing, arch-mixing is silently allowed.

## Follow-ons (deliberately not built)

- **PFSP** — weight pool sampling by live per-opponent win-rate instead of uniform
  (the `opponent_network_id` column + game results provide the data).
- **Learner-only plies** — filter league-game records to the learner's side (declined for
  v1; a ~2-line conditional at the `game.cc` record-append site if ever wanted).
- **Regression-panel gate** — candidate must not regress vs a frozen panel (the scoreboard
  side of the RPS fix; `main.go` TODO comment).
- Backend-dedup quirk (pre-existing): `ChesskersBackend` never sets its configuration hash,
  so `IsSameConfiguration` dedup never fires and every player/pool slot loads its own net
  copy. Harmless at current net sizes; fix in the fork if VRAM ever pinches.
