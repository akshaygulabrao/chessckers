# Run 9 — d6/e6/f6 vs pawn wall (unknown-answer), warm-started from run 8

> **PENDING LAUNCH** (set up 2026-06-29). The first **unknown-answer** experiment — we do NOT
> know if Black can force the win — so the gate + `cc strength` (built in run 8) finally earn
> their keep: self-play W/B balance is **not** a strength signal here, the gate's net-vs-net Elo is.

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `V6_d6e6f6` (proposed) |
| Start FEN | `8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] w - - 0 1` — 3× kk towers (d6/e6/f6) vs full pawn wall + Ke1, **White to move** |
| Arch | SE-ResNet v5 c48/b5 ~364K (same as run 8) |
| Optimizer | Adam lr=1e-3 (same) |
| Rules | v6 bottom-*d* charge (same) |
| **Init** | **WARM-START from run 8's net** (`~/chessckers-backups/run8-e8d8-v6gated-20260629/weights.pt`), not cold — transfer the learned mate |
| Gate | in-fleet lc0 gate live (calcElo > −20), 40-game candidate-vs-best |
| Fleet box | vast 42618148 (RTX 3060) — launched 2026-06-29 |
| Status | **active** — warm-started from run 8, self-play on d6/e6/f6 |

## Hypothesis

Two open questions:
1. **Is d6/e6/f6 even a Black win?** Unknown. Three 2-King towers must break a full pawn wall and
   mate Ke1 — much harder than the open e8/d8 KK-vs-K. A prior detour saw ~95/5 White and flat,
   consistent with *either* a deep cold-start trap *or* a genuinely White-favored position.
2. **Does the e8/d8 mate transfer?** Warm-starting from run 8's net (which solved e8/d8) tests
   whether the learned tower-coordination / charge-mate skills bootstrap progress here, vs a cold
   start that may never escape.

**Success = strength gain measured by the gate**, not self-play balance. On an unknown-answer
position, two copies of the same improving net can sit at any balance while both improve — so
`cc strength` (cumulative gate Elo over promotions) is the signal to watch, plus a rejection by
the gate is a real "this candidate is worse" flag.

## Design delta vs run 8

- **New start position** — `board.cc kStartposFen` → the d6/e6/f6 FEN (compiled into the fork; needs
  a rebuild). Verified it renders correctly through PyVariant.
- **Warm-start** from run 8's net instead of cold random init (the transfer-learning lever).
- Everything else identical: v5 c48/b5, Adam 1e-3, v6 charge rule, in-fleet gate, `cc strength`.

## Log

- `06-29` Set up: `board.cc` → d6/e6/f6 (White-to-move); run 8's net backed up off-box for the
  warm-start.
- `06-29` **Box rebooted (vast infra) ~12:58 → fleet died silently; restarted.** No crash/OOM —
  logs ran clean to step 1517 / gate match 57, then the box rebooted (booted 13:01) and tmux
  (server+trainer+client) was gone. All state persisted on disk, so **warm-resumed run 9 from
  `trainer/run1/weights.pt`** (not re-seeded) after deleting 13 reboot-truncated 0-byte chunks.
  The first `upload_network` 400 is the benign already-exists dedup.
- `06-29` **Auto-restart hardening added** (so a reboot self-heals): `restart_fleet.sh` (idempotent
  warm-resume relaunch) + an `@reboot` cron on the box + `cc restart` + fresh-run installs the cron.
  See [[chessckers-vast-reboot-autorestart]].
- `06-29` **Launched, warm-start verified.** `cc fresh-run --base=/workspace/run8_seed/weights.pt`
  (new `--base` flag). Trainer `base=/workspace/run8_seed/weights.pt` (NOT random init); gate
  bootstrap-promoted the warm net (SHA 585ea4f4…, ≠ the cold-init cf42568…). Self-play runs from
  d6/e6/f6 (White opens `g2g3`). **Early signal:** first game was a 10-ply Black win (`0-1`) — the
  warm net converts as Black rather than flailing near-random, so the e8/d8 mate appears to
  transfer. One game = noise; watch the gate Elo, not balance.

## Result

<unknown-answer — fill once it runs. Watch the gate Elo (`cc strength`), NOT the W/B balance.
Reject events = real regression signals. Open question: does Black break the pawn wall at all?>
