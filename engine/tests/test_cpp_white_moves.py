"""Slice 3b oracle test: C++ white_legal_moves vs the live Rust extension
(exact-ordered) and the pure-Python reference (set), over self-play rollouts +
hand-crafted edge positions (castling both forms, promotions, en passant, check).

White is FIDE pseudo-legal filtered by the CHESSCKERS check predicate, so
python-chess's own legality is NOT the oracle; the Rust extension is. Castling
emits both the e1g1 and the king-to-rook e1h1 form, which the port reproduces.
"""
from __future__ import annotations

import random

import chess
import pytest

from chessckers_engine.variant_py import moves_white as mw
from chessckers_engine.variant_py.client import PyVariantClient
from chessckers_engine.variant_py.state import STARTING_FEN, parse_fen

cpp = pytest.importorskip("chessckers_cpp")
rs = pytest.importorskip("chessckers_movegen")

ROLLOUT_SEEDS = [
    STARTING_FEN,                                                       # full White army
    "8/8/3kkk2/2PPPP2/2PPPP2/8/8/4K3[d6:kk,e6:kk,f6:kk] w - - 0 1",     # king + pawns vs towers
    "7p/8/8/8/8/8/8/4K3[h8:ssss] w - - 0 1",
]

EDGE_FENS = [
    "k7/8/8/8/8/8/8/R3K2R[a8:kk] w KQ - 0 1",          # castle both sides (4 castling dicts)
    "7k/4P3/8/8/8/8/8/4K3[h8:kk] w - - 0 1",           # pawn push promotions e7->e8
    "5k2/6P1/8/8/8/8/8/4K3[f8:kk,h8:kk] w - - 0 1",    # capture-promotion g7xf8/h8 + push
    "4k3/8/8/3pP3/8/8/8/4K3[d5:s] w - d6 0 1",         # en passant e5xd6
    "8/8/8/8/8/2k5/8/4K3[c3:kkk] w - - 0 1",           # White king restricted near a Black tower
]


def _wb(fen):
    b = parse_fen(fen).board
    stacks = {int(s): p for s, p in parse_fen(fen).stacks.items()}
    ep = -1 if b.ep_square is None else int(b.ep_square)
    return (
        int(b.occupied), int(b.occupied_co[chess.WHITE]), int(b.pawns), int(b.knights),
        int(b.bishops), int(b.rooks), int(b.queens), int(b.kings), int(b.castling_rights), ep,
        stacks,
    )


def _canon(m: dict) -> tuple:
    return (m["uci"], m["from"], m["to"], m["piece"], m.get("color"), m.get("capture"),
            m.get("promotion"))


def _collect(n_games=10, max_plies=40, seed=777):
    client = PyVariantClient()
    rng = random.Random(seed)
    fens = []
    for g in range(n_games):
        st = client.new_game(ROLLOUT_SEEDS[g % len(ROLLOUT_SEEDS)])
        for _ in range(max_plies):
            fens.append(st["fen"])
            legal = st.get("legalMoves") or []
            if st.get("status") or not legal:
                break
            st = client.make_move(st["fen"], rng.choice(legal)["uci"])
    return fens


ALL_FENS = _collect() + EDGE_FENS


def test_white_legal_moves_ordered_vs_rust():
    assert len(ALL_FENS) > 100
    for fen in ALL_FENS:
        args = _wb(fen)
        cpp_list = [_canon(m) for m in cpp.white_legal_moves(*args)]
        rs_list = [_canon(m) for m in rs.white_legal_moves(*args)]
        assert cpp_list == rs_list, f"\n fen={fen}\n cpp={cpp_list}\n  rs={rs_list}"


def test_white_legal_moves_setdiff_vs_python():
    saved = mw._rs_movegen
    mw._rs_movegen = None
    try:
        for fen in ALL_FENS:
            state = parse_fen(fen)
            if state.board.turn != chess.WHITE:
                continue
            args = _wb(fen)
            py_set = {_canon(m) for m in mw.white_legal_moves(state)}
            cpp_set = {_canon(m) for m in cpp.white_legal_moves(*args)}
            assert cpp_set == py_set, (
                f"\n fen={fen}\n only in cpp: {sorted(cpp_set - py_set)}"
                f"\n only in py:  {sorted(py_set - cpp_set)}"
            )
    finally:
        mw._rs_movegen = saved


@pytest.mark.parametrize("fen,expect", [
    (EDGE_FENS[0], {"e1g1", "e1h1", "e1c1", "e1a1"}),   # both castle forms present
    (EDGE_FENS[1], {"e7e8q", "e7e8r", "e7e8b", "e7e8n"}),  # promotions present
])
def test_edge_positions_contain_expected_ucis(fen: str, expect: set):
    ucis = {m["uci"] for m in cpp.white_legal_moves(*_wb(fen))}
    assert expect <= ucis, f"missing {expect - ucis} in {sorted(ucis)}"
