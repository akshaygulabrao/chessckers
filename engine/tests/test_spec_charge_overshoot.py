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
import pytest

import chessckers_engine.variant_py.client as _cl
import chessckers_engine.variant_py.moves_black as _mb
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

RUST = pytest.mark.skipif(_mb._rs_movegen is None, reason="Rust extension not built")


@pytest.fixture
def bypass_rust(monkeypatch):
    """Force the pure-Python move-gen / check path everywhere it is consulted."""
    monkeypatch.setattr(_mb, "_rs_movegen", None)
    monkeypatch.setattr(_cl, "_rs_movegen", None)
    monkeypatch.setattr(_mw, "_rs_movegen", None)


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


# ---- pure-Python path agrees (covers the spec-test / NO_RUST surface) ----

def test_pure_python_path(bypass_rust):
    rim = _rim_charge(AFTER_B)
    assert [m["uci"] for m in rim] == ["e2e0->e1"]
    r = PyVariantClient().make_move(AFTER_B, "e2e0->e1")
    assert parse_fen(r["fen"]).board.king(chess.WHITE) is None
    assert _mw._is_white_in_chessckers_check(parse_fen(AFTER_W)) is True


# ---- Python and Rust must agree byte-for-byte on this position ----

@RUST
def test_python_rust_charge_equivalence():
    """Every Python charge dict appears in the Rust legal-move list with byte-
    identical fields (no mandate is active here, so no charge is suppressed)."""
    st = parse_fen(AFTER_B)
    wk = st.board.king(chess.WHITE)
    rs_all = _mb._rs_movegen.all_black_legal_moves(
        st.board.occupied, st.board.occupied_co[chess.WHITE],
        -1 if wk is None else wk, st.stacks,
    )
    rs = {m["uci"]: (m["to"], m["capture"], tuple(m["waypoints"] or ())) for m in rs_all}
    for m in _mb.black_charge_moves(st):
        fields = (m["to"], m["capture"], tuple(m["waypoints"] or ()))
        assert rs.get(m["uci"]) == fields, m["uci"]
    # The rim charge is present and identical in both engines; the bare form is
    # absent from both.
    assert rs.get("e2e0->e1") == ("e1", "e1", ("e0",))
    assert "e2e1" not in rs
