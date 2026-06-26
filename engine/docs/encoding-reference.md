# Encoding reference (run-independent)

Stable arch/encoding tables shared across runs. Per-run docs (`runs/runN.md`) link
here instead of re-pasting these. Update this file when the **encoding itself**
changes (and note the change in the run doc that introduced it).

## Position tensor (15 channels, 8×8)

| Ch | Content |
|---|---|
| 0–5 | White pieces (P,N,B,R,Q,K) — one-hot bitboard |
| 6 | Stone-top (Black-Pawn bitboard = Stone tower top) |
| 7 | King-top (Black-King bitboard = King tower top) |
| 8–12 | **Per-depth stack** (v5+): `s=0.33`, `S=0.67`, `k=1.0`, `none=0`. Channel `8+d` = piece at stack position `d` (bottom-to-top). |
| 13 | Side to move (all-1 if Black, all-0 if White) |
| 14 | Rank-8 win counter (r8 / 3) |

> **Pre-v5** channels 8–12 were *aggregate* (tower_height, stone_count, king_count,
> top_is_unmoved_stone, second_is_king), all `/24`. v5 replaced them with the
> per-depth channels above and rescaled denominators to `/MAX_TOWER_HEIGHT` (=5).
> Channel count is unchanged (POS_C=15 / POS_C_V2=16).

## Move feature vector (240-dim)

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

## V2/V4/V5 position tensor (16 channels, 10×10)

Same 15 channels written into the 8×8 interior of a 10×10 grid, plus:

| Ch | Content |
|---|---|
| 15 | On-board mask (1 on the 8×8 interior, 0 on the rim ring) |
