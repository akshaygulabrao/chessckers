"""Slice 2d oracle test: the FULL Black legal move list (mandate applied) and the
§4 mandate trigger.

This is the capstone for Black move-gen. all_black_legal_moves is verified
EXACT-ORDERED against the live Rust extension (the authoritative move order the
policy head indexes) and as a canonical SET against the pure-Python reference
moves_black._all_black_legal. The corpus mixes mandate-active positions (only
capturing moves survive) and mandate-inactive ones (full quiet+deploy+charge+
chain list).
"""
from __future__ import annotations

import chess
import pytest

from chessckers_engine.variant_py import moves_black as mb
from chessckers_engine.variant_py.state import STARTING_FEN, parse_fen

cpp = pytest.importorskip("chessckers_cpp")

START_BLACK = STARTING_FEN.replace(" w ", " b ")

CORPUS = [
    "8/8/3kkk2/2PPPP2/2PPPP2/8/8/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",   # mandate active -> captures only
    "8/8/3kkk2/8/8/8/8/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",            # no captures -> full list
    START_BLACK,                                                     # full board, mandate inactive
    "8/8/8/8/4k3/8/8/4K3[e4:kkk] b - - 0 1",                        # charges + quiets + deploys
    "k7/1N6/2N5/3N4/4N3/5N2/8/4K3[a8:kkkkk] b - - 0 1",             # mandate active -> long chain
    "8/8/8/8/3k4/2P5/2P5/4K3[d4:kkk] b - - 0 1",                    # mandate active (diag capture)
    "8/8/8/8/4k3/8/4P3/4K3[e4:kkk] b - - 0 1",                      # inactive: capturing charge in full list
]


def _bb(state):
    occ = int(state.board.occupied)
    occw = int(state.board.occupied_co[chess.WHITE])
    wk = state.board.king(chess.WHITE)
    king_sq = -1 if wk is None else int(wk)
    stacks = {int(s): p for s, p in state.stacks.items()}
    return occ, occw, king_sq, stacks


def _canon(m: dict) -> tuple:
    def t(v):
        return tuple(v) if v is not None else None

    return (
        m["uci"], m["from"], m["to"], m["piece"], m.get("color"), m.get("capture"),
        t(m.get("waypoints")), t(m.get("chainHops")), m.get("promotion"),
        t(m.get("demotedKings")), m.get("demotionsRequired"), t(m.get("sourceKingPositions")),
        m.get("deployCount"), t(m.get("_chain_all_captures")), m.get("cadence"),
        m.get("_is_suicide"), m.get("_chain_promotes"),
    )


def _py_all_black(state):
    saved = mb._rs_movegen
    mb._rs_movegen = None
    try:
        return mb._all_black_legal(state)
    finally:
        mb._rs_movegen = saved


@pytest.mark.parametrize("fen", CORPUS)
def test_all_black_ordered_vs_rust(fen: str):
    rs = mb._rs_movegen
    if rs is None:
        pytest.skip("rust extension not built")
    state = parse_fen(fen)
    occ, occw, king_sq, stacks = _bb(state)
    rs_list = [_canon(m) for m in rs.all_black_legal_moves(occ, occw, king_sq, stacks)]
    cpp_list = [_canon(m) for m in cpp.all_black_legal_moves(occ, occw, king_sq, stacks)]
    assert cpp_list == rs_list, f"\n cpp={cpp_list}\n  rs={rs_list}"


@pytest.mark.parametrize("fen", CORPUS)
def test_all_black_setdiff_vs_python(fen: str):
    state = parse_fen(fen)
    occ, occw, king_sq, stacks = _bb(state)
    py_set = {_canon(m) for m in _py_all_black(state)}
    cpp_set = {_canon(m) for m in cpp.all_black_legal_moves(occ, occw, king_sq, stacks)}
    assert cpp_set == py_set, (
        f"\n only in cpp: {sorted(cpp_set - py_set)}\n only in py:  {sorted(py_set - cpp_set)}"
    )


@pytest.mark.parametrize("fen", CORPUS)
def test_mandate_matches_both_oracles(fen: str):
    state = parse_fen(fen)
    occ, occw, _king, stacks = _bb(state)
    cpp_m = cpp.black_mandatory_capture_active(occ, occw, stacks)
    py_m = mb.black_mandatory_capture_active(state)
    assert cpp_m == py_m, fen
    rs = mb._rs_movegen
    if rs is not None:
        assert cpp_m == rs.black_mandatory_capture_active(occ, occw, stacks), fen


def test_corpus_covers_both_mandate_branches():
    actives = []
    for fen in CORPUS:
        occ, occw, _k, stacks = _bb(parse_fen(fen))
        actives.append(cpp.black_mandatory_capture_active(occ, occw, stacks))
    assert any(actives) and not all(actives), "corpus must exercise mandate on AND off"
