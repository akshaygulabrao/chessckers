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


# --- apply-path regression: a diagonal chain that lands on the origin's rank ---
#
# Provenance: an actual fleet self-play game (2026-06-10) ended with a PHANTOM
# king capture. The losing move was the diagonal capture chain c2:c3~e1~g3->g3.
# Its path is c3-d2-e1-f2-g3 (capturing the Pawns on d2 and f2); the White King
# on e3 is NOT on that path. But because the chain's origin (c3) and final
# landing (g3) share rank 3, apply_black_move_known mis-dispatched it to the
# orthogonal Charge apply, which walked the straight line c3-d3-e3-f3-g3 and
# captured the king on e3 — ending the game as a bogus Black win. The dispatch
# now checks chainHops before _is_orthogonal_move. These tests lock that in.

def _make(fen: str, uci: str):
    return PyVariantClient().make_move(fen, uci)


def test_chain_landing_on_origin_rank_does_not_capture_offpath_king():
    """The minimal repro: chain c3~e1~g3 (lands on rank 3, like its origin)
    must capture ONLY its path Pawns (d2, f2) and leave the off-path King on
    e3 alone — the game continues, it is NOT a Black win."""
    fen = "8/8/1p2p3/4P3/6P1/2p1K3/PP1P1P1P/8[c3:S,b6:SS,e6:kkS] b - - 0 1"
    r = _make(fen, "c2:c3~e1~g3->g3")
    board = r["fen"].split("[")[0]
    rank3 = board.split("/")[5]   # ranks are listed 8..1; index 5 == rank 3
    rank2 = board.split("/")[6]
    assert "K" in rank3, "white king on e3 must survive the chain"
    assert rank2 == "PP5P", f"d2 and f2 must be captured, got rank2={rank2!r}"
    assert r.get("status") is None and r.get("winner") is None, \
        "chain capturing only Pawns must not end the game"


def test_fleet_game_2026_06_10_no_phantom_king_capture():
    """Full replay of the offending fleet game from the simplified start. The
    final move must NOT win for Black (the king is never on the chain's path);
    the game continues with White to move."""
    start = "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] w - - 0 1"
    moves = ("c2c4 d6b6 g2g3 e6d5[1] e2e3 c2:d5~b3->b3 g3g4 e6d5 e1e2 d5e6 "
             "e3e4 f6e6{2} e2e3 b3c3 e4e5 c2:c3~e1~g3->g3").split()
    c = PyVariantClient()
    r = c.new_game(start)
    for mv in moves:
        r = c.make_move(r["fen"], mv)
    assert r.get("status") is None, "game must NOT have ended on the chain move"
    assert r["turn"] == "white", "White is to move after the chain"
    assert "K" in r["fen"].split("[")[0].split("/")[5], "white king must survive"
