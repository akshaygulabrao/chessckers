"""A King-top tower can check/capture a White king on a board edge by charging
*past* it to the rim.

The bug: a charge that overshoots the king onto a rim square (capturing the
king in transit, then falling back) was (a) mis-applied as a ram — landing on
the king, which does NOT capture it — so the king survived, and (b) missed by
the check predicate, which required an *on-board* landing past the king. Both
the Python path and the Rust mirror had it.

Concretely the d3/e3 curriculum seed is a forced **mate-in-1**: Black plays
`d3e2`, leaving White's king on e1 with a kk tower behind it on e2. White is in
Chessckers-check because `e2` can charge the rim square `e0`, capturing the king
on `e1` on the way (`e2e0->e1`). A ram landing on e1 would NOT capture — the
rim-overshoot is what makes it a capture, and the notation must say so.
"""
from __future__ import annotations

import chess

import chessckers_engine.variant_py.moves_white as _mw
import endgame_solver as es
from chessckers_engine.variant_py import PyVariantClient
from chessckers_engine.variant_py.state import parse_fen

# d3/e3 seed, Black to move.
SEED = "8/8/8/8/8/3kk3/8/4K3[d3:kk,e3:kk] b - - 0 1"
# After d3e2: White king e1, Black kk on e2 (behind it) and e3. The diagnostic
# position for the charge — given here Black-to-move to probe the king-capture.
AFTER_B = "8/8/8/8/8/4k3/4k3/4K3[e2:kk,e3:kk] b - - 0 1"
# Same board, White to move — the real in-game position after d3e2.
AFTER_W = "8/8/8/8/8/4k3/4k3/4K3[e2:kk,e3:kk] w - - 0 1"


def _rim_charge(fen):
    """The single rim-overshoot charge that captures e1, from AFTER_B."""
    moves = PyVariantClient().new_game(fen)["legalMoves"]
    return [m for m in moves if "->" in m["uci"]]


# ---- notation + generation ----

def test_rim_charge_notation_and_fields():
    """The overshoot charge is spelled `e2e0->e1` and carries the rim key in
    waypoints — never the bare `e2e1` (which would read as a ram)."""
    rim = _rim_charge(AFTER_B)
    assert len(rim) == 1, rim
    m = rim[0]
    assert m["uci"] == "e2e0->e1"
    assert m["to"] == "e1"
    assert m["capture"] == "e1"          # the king, captured in transit
    assert m["waypoints"] == ["e0"]      # aimed at the rim square e0
    # The ambiguous bare form must NOT appear as a legal move.
    ucis = {x["uci"] for x in PyVariantClient().new_game(AFTER_B)["legalMoves"]}
    assert "e2e1" not in ucis


# ---- apply (capture) ----

def test_rim_charge_captures_the_king():
    r = PyVariantClient().make_move(AFTER_B, "e2e0->e1")
    assert parse_fen(r["fen"]).board.king(chess.WHITE) is None
    assert (r["status"], r["winner"]) == ("variantEnd", "black")


# ---- check / mate detection ----

def test_white_is_in_check_after_d3e2():
    g = PyVariantClient().make_move(SEED, "d3e2")
    assert g["check"] is True
    assert (g["status"], g["winner"]) == ("mate", "black")


def test_attack_predicate_sees_the_charge():
    st = parse_fen(AFTER_W)
    king_sq = st.board.king(chess.WHITE)
    assert _mw._square_attacked_by_black_chessckers(st, king_sq) is True


# ---- the headline: the seed is a forced mate-in-1 ----

def test_seed_is_mate_in_one():
    assert es.distance_to_mate(SEED, 9) == 1
    assert es.best_black_moves(SEED, 9) == ["d3e2"]


# ---- pure-Python path ----

def test_pure_python_path():
    rim = _rim_charge(AFTER_B)
    assert [m["uci"] for m in rim] == ["e2e0->e1"]
    r = PyVariantClient().make_move(AFTER_B, "e2e0->e1")
    assert parse_fen(r["fen"]).board.king(chess.WHITE) is None
    assert _mw._is_white_in_chessckers_check(parse_fen(AFTER_W)) is True
