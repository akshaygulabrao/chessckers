"""Check/checkmate detection must see Black's chain & overshoot king-captures.

The targeted attack model missed multi-hop diagonal chains (and off-grid
overshoots): Black can capture the White king by turning a chain around, so a
single-diagonal scan reports "not in check" when the king is in fact capturable.
That let White play into a king-capture and missed checkmates. The fix defines
check via the real capture generator.

Runs the pure-Python path (the Rust mirror still has the pre-§3B rule).
"""
from __future__ import annotations

import pytest

import chessckers_engine.variant_py.client as _cl
import chessckers_engine.variant_py.moves_black as _mb
import chessckers_engine.variant_py.moves_white as _mw
from chessckers_engine.variant_py import PyVariantClient
from chessckers_engine.variant_py.state import parse_fen

# Black f4:ss tower; White Pawn e3; White King e1. Black can capture the King
# ONLY via a chain: f4->d2 (capture e3), turn, d2->f0 (capture King e1). No
# single diagonal from f4 reaches e1.
CHAIN_CHECK_FEN = "8/8/8/8/5p2/4P3/8/4K3[f4:ss] w - - 0 1"


@pytest.fixture(autouse=True)
def _bypass_rust(monkeypatch):
    monkeypatch.setattr(_mb, "_rs_movegen", None)
    monkeypatch.setattr(_cl, "_rs_movegen", None)


def test_chain_threat_is_detected_as_check():
    assert _mw._is_white_in_chessckers_check(parse_fen(CHAIN_CHECK_FEN)) is True
    assert PyVariantClient().new_game(CHAIN_CHECK_FEN)["check"] is True


def test_white_cannot_ignore_a_chain_check():
    """Every legal White move must resolve the chain check — none may leave the
    King capturable next turn."""
    c = PyVariantClient()
    g = c.new_game(CHAIN_CHECK_FEN)
    assert g["legalMoves"], "white should have check-resolving moves, not stalemate"
    for m in g["legalMoves"]:
        after = parse_fen(c.make_move(CHAIN_CHECK_FEN, m["uci"])["fen"])
        assert not _mw._is_white_in_chessckers_check(after), \
            f"{m['uci']} leaves the King capturable"
