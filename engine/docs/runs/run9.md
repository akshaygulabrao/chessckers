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
| Status | pending launch |

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
  warm-start. Pending launch.

## Result

<unknown-answer — fill once it runs. Watch the gate Elo (`cc strength`), NOT the W/B balance.
Reject events = real regression signals. Open question: does Black break the pawn wall at all?>
