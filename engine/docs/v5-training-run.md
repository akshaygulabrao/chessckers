# Chessckers V5 Training Run

## What changed from V4

### 1. Input encoding — per-depth tower channels

**V4** used 5 aggregate channels for Black towers:

| Ch | Plane | Encoding |
|---|---|---|
| 8 | tower_height | height / 24 |
| 9 | stone_count | count / 24 |
| 10 | king_count | count / 24 |
| 11 | top_is_unmoved_stone | 1 if top = `s` |
| 12 | second_is_king | 1 if stack[-2] = `k` |

This gave the net aggregate statistics but threw away the **order** of pieces
inside the tower. A tower `"kSs"` (King bottom, moved Stone middle, unmoved
Stone top) and a tower `"skS"` (Stone bottom, King middle, moved Stone top)
looked identical in channels 8–12 — same height, same counts, same top marker.

**V5** replaces these with 5 **per-depth channels** (8–12), where channel
`8 + d` is the piece at stack position `d` (bottom-to-top):

| Value | Piece |
|---|---|
| `0.00` | empty (no piece at this depth) |
| `0.33` | unmoved Stone (`s`) |
| `0.67` | moved Stone (`S`) |
| `1.00` | King (`k`) |

A tower `"kSs"` on e7 writes:

```
ch 8 (stack[0]): 1.00   ← bottom = k
ch 9 (stack[1]): 0.67   ← middle = S
ch10 (stack[2]): 0.33   ← top = s
ch11 (stack[3]): 0.00   ← beyond height
ch12 (stack[4]): 0.00
```

The full order is now visible. Height, stone/king counts, and top/second
identity are all recoverable from these channels — no information is lost,
and the network can learn positional relationships between pieces in a tower.

**Channel count unchanged:** POS_C=15 (V1) / POS_C_V2=16 (V2/V4/V5). The 5
aggregate channels were simply replaced by 5 per-depth channels.

### 2. Tower height cap at 5

A tower may never exceed 5 pieces. Friendly merges (quiet landing, deploy merge,
charge merge, capture-hop landing onto a friendly tower) that would produce a
tower taller than 5 are illegal. Enforced in:

- Python rules oracle (`variant_py/moves_black.py`)
- C++ engine (`akshay-chessckers-0/src/chessckers/movegen.hpp`)
- FEN parser (both Python and C++)

The cap simplifies the per-depth encoding (5 planes, no ambiguity) and prevents
pathological deep stacks that the network struggles to represent. The starting
position has 24 singleton towers — this rule only affects midgame merges.

All denominator constants changed from `/24.0` to `/MAX_TOWER_HEIGHT` (i.e.
`/5.0`) — height, stone_count, king_count on old nets; deploy_count in move
features.

### 3. Optimizer — Adam replacing SGD+Nesterov

**V4:** `torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, nesterov=True)`

**V5:** `torch.optim.Adam(model.parameters(), lr=1e-3)`

The switch from SGD+Nesterov to Adam is the main training-dynamics change.
V4's SGD used a high LR (0.02) with Nesterov momentum (0.9), scaled ~20× above
typical Adam LRs. V5 uses the standard Adam LR (1e-3) with its built-in
adaptive per-parameter learning rates.

This was the revert in commit `a872b5f` — the earlier optimizer (pre-V4) was
Adam, and SGD was an experiment. V5 returns to Adam.

### 4. Network tag

The `.arch.json` sidecar now writes `"version": "v5"`. The `build_model()`
function and `train_continuous` accept `--arch-version v5`. The architecture
is still `ChesskersScorerV2` (square-grounded gather head + SE-ResNet blocks) —
the version bump reflects the changed input encoding, not a trunk change.

## Run parameters

| Parameter | Value |
|---|---|
| Start position | `3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] w - - 0 1` (two kk towers on d8/e8, K on e1) |
| Architecture | V5 (SE-ResNet gather head, c_filters=48, n_blocks=5, d_hidden=256, se_ratio=8) |
| Params | ~364K |
| Optimizer | Adam, lr=1e-3 |
| Replay window | 400→4000 games (growing sublinear ramp, α=0.75) |
| Min buffer | 200 games |
| Buffer cap | 500,000 positions |
| Batch size | 1024 |
| Publish cadence | every 100 games |
| EMA decay | 0.999 |
| Value discount | 1.0 (off — WDL from moves-left head instead) |
| Value Q-ratio | 0.5 (50/50 z/q blend for variance reduction) |
| Replay factor | 8× throttle |
| Self-play visits | 800 |
| Self-play temperature | 1.0, decay over 15 moves |
| Dirichlet noise | ε=0.25, α=0.3 |

## Encoding reference

### Position tensor (15 channels, 8×8)

| Ch | Content |
|---|---|
| 0–5 | White pieces (P,N,B,R,Q,K) — one-hot bitboard |
| 6 | Stone-top (Black-Pawn bitboard = Stone tower top) |
| 7 | King-top (Black-King bitboard = King tower top) |
| 8–12 | **Per-depth stack**: `s=0.33`, `S=0.67`, `k=1.0`, `none=0` |
| 13 | Side to move (all-1 if Black, all-0 if White) |
| 14 | Rank-8 win counter (r8 / 3) |

### Move feature vector (240-dim)

| Bits | Feature |
|---|---|
| 0–63 | from_square one-hot |
| 64–127 | to_square one-hot |
| 128 | is_capture |
| 129 | is_chain (waypoints non-empty) |
| 130 | is_deploy |
| 131 | is_ortho (demotionsRequired set) |
| 132 | chain_length / 8 |
| 133 | deploy_count / MAX_TOWER_HEIGHT |
| 134 | demotions_required / 8 |
| 135–139 | promotion piece one-hot |
| 140–239 | waypoint mask (10×10 grid flat, 10 cells each) |

### V2/V4/V5 position tensor (16 channels, 10×10)

Same 15 channels written into the 8×8 interior of a 10×10 grid, plus:

| Ch | Content |
|---|---|
| 15 | On-board mask (1 on the 8×8 interior, 0 on the rim ring) |

## Files changed for V5

| Layer | File | Change |
|---|---|---|
| Spec | `chessckers.md` | §1 Max height rule, v3→v5 |
| Python | `encoding.py` | Per-depth channels 8–12, /24→/5 denominators |
| Python | `state.py` | MAX_TOWER_HEIGHT=5, FEN validation |
| Python | `moves_black.py` | Merge guards (quiet, deploy, charge, sprint) |
| Python | `model.py` | Version tag → "v5" |
| Python | `train_continuous.py` | Accept v5 arch; optimizer = Adam |
| C++ | `board.hpp` | MAX_TOWER_HEIGHT=5, FEN validation |
| C++ | `encode.hpp` | Per-depth channels, /24→/5 |
| C++ | `movegen.hpp` | Merge guards |
| C++ | `board.cc` | New start FEN (e8/d8 seed) |
| Bridge | `trainer_bridge.py` | Accept v5 arch |
