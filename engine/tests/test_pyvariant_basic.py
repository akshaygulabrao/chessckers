"""Basic PyVariantClient unit tests (no engine dependency).

These are the non-scala tests salvaged from the old test_pyvariant_diff.py.
The live-scalachess differential suite was retired: matching the (now
deprecated and, on the new §3B rules, incomplete) Scala engine exactly is no
longer the goal. PyVariant + the Rust extension are the authority. Correctness
coverage now lives in:
  - tests/test_spec_black_chains.py — hand-verified §3B chain/capture cases
  - tests/test_spec_offgrid_settle.py — off-grid overshoot + optional stops
  - tests/test_spec_check_detection.py — chain/overshoot king-capture checks
"""
from __future__ import annotations

import pytest

from chessckers_engine.variant_py import PyVariantClient

INITIAL_FEN_KQ = (
    "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"
    "[a6:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,"
    "a7:k,b7:k,c7:k,d7:k,e7:k,f7:k,g7:k,h7:k,"
    "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] w KQ - 0 1"
)


def test_pyvariant_constructs():
    client = PyVariantClient()
    client.close()


def test_unported_stubs_raise_not_implemented():
    """Interactive chain-stepping (UI-only) is not ported."""
    client = PyVariantClient()
    with pytest.raises(NotImplementedError):
        client.chain_step("fen", "f6", [])
    with pytest.raises(NotImplementedError):
        client.chain_end("fen", "f6", [])


def test_new_game_default_returns_starting_position():
    state = PyVariantClient().new_game()
    assert state["turn"] == "white"
    assert state["status"] is None
    assert state["winner"] is None
    assert "RNBQKBNR" in state["fen"]
    # 24 stacks in the overlay, each "square:pieces" -> 24 colons.
    assert state["fen"].count(":") == 24


def test_new_game_with_fen_echoes_input():
    """For non-default input FENs, the returned `fen` is the input verbatim."""
    state = PyVariantClient().new_game(INITIAL_FEN_KQ)
    assert state["fen"] == INITIAL_FEN_KQ
