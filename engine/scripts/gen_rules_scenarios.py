#!/usr/bin/env python3
"""Generate a *curated* PyVariant oracle corpus of rules-critical scenarios.

Companion to gen_parity_corpus.py. That script plays deterministic random
games and is excellent at breadth — it floods the C++ parity test with the
common Black/White move shapes (quiet diagonals, deploys, cadence chains,
charges) at high volume. But random 80-ply games almost never *reach* the rare
corners that decide correctness: the §3B worked examples (cadence lock, rim
landing, off-grid overshoot), the §4 capture Mandate, the §5 win conditions
(rank-8 hold, king capture, Black self-stalemate), promotion across rank 1,
and the opening double-move. (Audit of the 8000-line random corpus: only 7
terminal positions, 0 carrying the rank-8 `{r8:..}` counter, 1 carrying the
opening `{wm:2}`.)

This script enumerates a hand-picked FEN per rule corner — analogous to how
lc0's board_test.cc curates Kiwipete / Position3-6 rather than relying solely
on the startpos perft — and, for the scenarios whose *point* is a transition
(reach rank 3 of the r8 counter, capture the king, promote), also plays the
decisive move and records the resulting position. Output is the SAME JSONL
schema as the random corpus, so the C++ parity path consumes it unchanged:

    {"fen": ..., "legal": [uci, ...sorted], "status": ..., "winner": ...,
     "apply": {uci: resulting_fen, ...}}

The `apply` map (PyVariant's resulting FEN per legal move) is the extra hook the
random corpus lacks: the C++ rules_scenarios_test.cc asserts
serialize_fen(apply_native(parse_fen(fen), m)) matches it for every move,
exercising the *apply* path (promotion, charge demotion, captures, the
{wm}/{r8} turn-state machine) — which the move-gen/status parity test never
touches. It is omitted on terminal positions (no legal moves).

Regenerate after any rule change (alongside gen_parity_corpus.py):

    .venv/bin/python scripts/gen_rules_scenarios.py \
        ../akshay-chessckers-0/src/chessckers/corpus/rules_scenarios.jsonl

Determinism: no RNG — every position is either an explicit seed FEN or the
result of an explicit named move, so the corpus is fully reproducible.
"""
from __future__ import annotations

import json
import sys

from chessckers_engine.variant_py import PyVariantClient

# (label, fen) — a curated position per rule corner. `label` is for the
# generator's own diagnostics only; it is not written to the corpus.
# Glyph reminder: board portion is python-chess (Black Stone = 'p', Black King
# = 'k', White uppercase); the [..] overlay carries each tower bottom-to-top
# ('s'=unmoved stone, 'S'=moved stone, 'k'=king).
SEEDS: list[tuple[str, str | None]] = [
    # ---- §3B diagonal captures: the two spec worked examples ----
    # g3 King-top tower, Rook h2, King a1: the SAME on-grid keys produce two
    # distinct hops — c2 (rim landing on i1) and c3 (off-grid overshoot past
    # i1) — both falling back to h2. Cadence is the discriminator.
    ("worked_ex1_rim_and_overshoot", "8/8/8/8/8/6k1/7R/K7[g3:sk] b - - 0 1"),
    # f5 King-top tower, Rook h3, Knight h1, King e1: cadence-3 chain that
    # overshoots off-grid on its 2nd hop and *settles* on the last board square
    # (h1), capturing Rook + Knight.
    ("worked_ex2_cadence3_settle", "8/8/8/5k2/8/7R/8/4K2N[f5:sk] b - - 0 1"),
    # ---- §3A non-capturing moves ----
    ("quiet_stone_forward_only", "8/8/4p3/8/8/8/8/4K3[e6:s] b - - 0 1"),
    ("quiet_king_any_diag_h2", "8/8/8/4k3/8/8/8/4K3[e5:kk] b - - 0 1"),
    ("deploy_height3", "8/8/8/4k3/8/8/8/4K3[e5:kkk] b - - 0 1"),
    ("sprint_rank8", "3p4/8/8/8/8/8/8/4K3[d8:s] b - - 0 1"),
    # ---- §3C charges (orthogonal; free path captures, King demotions paid) ----
    ("charge_file_capture", "8/8/8/8/8/8/4P3/4K3[e5:kk] b - - 0 1"),
    # height-3 King tower => charges offer the demoted-King choice suffix {a,b}.
    ("charge_demotion_choice", "8/8/8/4k3/8/8/8/4K3[e5:kkk] b - - 0 1"),
    # ---- §4 Mandate: an available normal-landing capture suppresses quiets ----
    ("mandate_forces_capture", "8/8/8/8/8/5p2/6k1/4K3[f5:s,g4:s] b - - 0 1"),
    # ---- ram landing on the King does NOT capture it (only path Whites) ----
    ("ram_lands_on_king", "8/8/8/8/8/5k2/4P3/3K4[f3:sk] b - - 0 1"),
    # ---- §5 promotion: a Black move whose path touches rank 1 promotes every
    # Stone in the tower to a King. Height-2 Stone tower on c3 sliding to a1
    # (White king parked on h8, off the diagonal, so nothing is captured).
    ("promo_cross_rank1", "7K/8/8/8/8/2p5/8/8[c3:ss] b - - 0 1"),
    # ---- §5 win conditions (terminal seeds) ----
    # White in Chessckers-check (two King towers charge-threaten its king) with
    # no escape => mate / Black wins.
    ("white_mated", "7K/5kk1/8/8/8/8/8/8[f7:kk,g7:kk] w - - 0 1"),
    # White stalemate is a DRAW (asymmetric): no legal move, not in check.
    ("white_stalemate_draw", "7K/5k2/8/8/8/8/8/8[f7:kkk] w - - 0 1"),
    # Black self-stalemate is a LOSS: Black has no legal move => White wins.
    ("black_self_stalemate", "8/8/8/8/7K/2P5/1P6/p7[a1:s] b - - 0 1"),
    # rank-8 counter sitting at 2 (one more White turn on rank 8 wins).
    ("rank8_counter_two", "4K3/8/8/8/8/8/8/4k3[a1:k] w - - 0 1 {r8:2}"),
    # ---- opening double-move (the start position carries {wm:2}) ----
    ("opening_double_move", None),
]

# (label, fen, uci) — record the position *after* the named move, so terminal /
# win states that the static seeds only set up actually appear in the corpus.
TRANSITIONS: list[tuple[str, str | None, str]] = [
    # rank-8 win: from r8:2, the White king stays on rank 8 (e8->d8) => r8 hits
    # 3 => variantEnd / white.
    ("rank8_win_reached", "4K3/8/8/8/8/8/8/4k3[a1:k] w - - 0 1 {r8:2}", "e8d8"),
    # rank-8 reset: instead the king leaves rank 8 (e8->e7) => counter clears,
    # game continues.
    ("rank8_reset", "4K3/8/8/8/8/8/8/4k3[a1:k] w - - 0 1 {r8:2}", "e8e7"),
    # king capture in transit: a height-3 King tower charges e4->e1, its path
    # crossing the White king on e2 => variantEnd / black.
    ("king_captured_in_transit", "8/8/8/8/4k3/8/4K3/8[e4:kkk] b - - 0 1", "e4e1"),
    # promotion result: the c3 Stone tower slides to a1; both Stones promote, so
    # the resting tower is King-top (overlay 'kk') and the game continues.
    ("promotion_applied", "7K/8/8/8/8/2p5/8/8[c3:ss] b - - 0 1", "c3a1"),
    # opening double-move: after White's first sub-move the FEN still carries
    # {wm:1}-implied state and it remains White to move for the 2nd sub-move.
    ("opening_after_first_submove", None, "e2e4"),
]


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "rules_scenarios.jsonl"
    client = PyVariantClient()
    seen: set[str] = set()
    records: list[dict] = []

    def record(d: dict) -> None:
        fen = d["fen"]
        if fen in seen:
            return
        seen.add(fen)
        rec: dict = {
            "fen": fen,
            "legal": sorted(m["uci"] for m in d["legalMoves"]),
            "status": d["status"] or "",
            "winner": d["winner"] or "",
        }
        # Apply-parity oracle: PyVariant's resulting FEN for every legal move.
        # Skipped on terminal positions (none to apply).
        if d["legalMoves"]:
            rec["apply"] = {
                m["uci"]: client.make_move(fen, m["uci"])["fen"]
                for m in d["legalMoves"]
            }
        records.append(rec)

    for _label, fen in SEEDS:
        record(client.new_game(fen) if fen else client.new_game())

    for label, fen, uci in TRANSITIONS:
        start = client.new_game(fen) if fen else client.new_game()
        legal_ucis = {m["uci"] for m in start["legalMoves"]}
        if uci not in legal_ucis:
            print(f"error: transition {label!r} move {uci!r} not legal in "
                  f"{start['fen']!r}", file=sys.stderr)
            return 1
        record(start)  # the pre-move position is itself worth asserting
        record(client.make_move(start["fen"], uci))

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")

    n_terminal = sum(1 for r in records if r["status"])
    print(f"wrote {len(records)} unique scenario positions "
          f"({n_terminal} terminal) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
