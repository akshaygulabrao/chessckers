"""Spec-verified black capture move-gen — rebased from the live-scala diff tests.

Each expectation here was produced by PyVariant (pure-Python path) and then
hand-verified against chessckers.md §3B with the terminal renderer, NOT copied
from scalachess. Several encode cases scala got wrong or couldn't express
(off-grid overshoot, optional stops, cadence-distinguished landings, the n+1
landing slot), so they are ground truth independent of the deprecated engine.

Positions are chosen so the capture mandate fires — the listed captures are the
COMPLETE legal-move set, which lets us assert exact equality.
"""
from __future__ import annotations

from chessckers_engine.variant_py import PyVariantClient


def _ucis(fen: str) -> list[str]:
    return sorted(m["uci"] for m in PyVariantClient().new_game(fen)["legalMoves"])


def test_king_capture_jumps_to_n_plus_1_slot():
    """A height-1 Stone on b5 with the White King on its forward diagonal at
    c4 (d=1) must JUMP it and land on the n+1 slot d3 — not land on c4. (The
    old diff test wrongly expected `b5c4`.) Mandate fires, so this is the only
    move."""
    assert _ucis("8/8/8/1p6/2K5/8/8/8[b5:s] b - - 0 1") == ["c2:b5~d3->d3"]


def test_chain_optional_stop_and_continue_to_king():
    """f4 height-2 Stone tower; mandate fires (White Pawn on e3, d=1, on the
    forward-left diagonal). Capturing e3:
      - cadence 2 -> land d2 (stop), OR continue: d2->f0 (rim) capturing the
        King on e1, falling back to e1;
      - cadence 3 -> land c1 (on rank 1, promotes).
    """
    assert _ucis("8/8/8/6P1/5p2/4P3/8/4K3[f4:ss] b - - 0 1") == [
        "c2:f4~d2->d2",        # stop after capturing e3
        "c2:f4~d2~f0->e1",     # continue: capture the King, rim-fallback to e1
        "c3:f4~c1->c1",        # cadence 3, lands on rank 1 (promotes)
    ]


def test_chain_cadence_distinguishes_landings_to_same_square():
    """e4 height-3 Stone tower; mandate fires (White Pawn on d3). Capturing d3
    down-left: cadence 2 -> c2; cadence 3 -> b1 (rank 1); cadence 4 -> overshoot
    to a0 (rim) then fall back to b1. The cadence-3 and cadence-4 moves both
    rest on b1 but are distinct candidates, told apart by the leading cadence."""
    assert _ucis("8/8/8/5P2/4p3/3P4/8/4K3[e4:sss] b - - 0 1") == [
        "c2:e4~c2->c2",
        "c3:e4~b1->b1",
        "c4:e4~a0->b1",
    ]
