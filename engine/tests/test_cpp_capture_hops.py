"""Slice 1 oracle test: the C++ §3B capture atom (`find_capture_hops`) must match
the PyVariant reference `_find_capture_hops` field-for-field AND in emit order.

Oracle: chessckers_engine.variant_py.moves_black._find_capture_hops (pure Python).
C++:    chessckers_cpp.find_capture_hops.

For each position we EXHAUSTIVELY sweep every start square (0..63) × every
diagonal direction × n in 1..8, isolating the hop atom before chains compound
any bug. This is the "field-level diff before chains" gate the port roadmap
calls for. Hops are compared as ordered tuples, so a difference in count,
order, captures, cadence, overshoot/suicide flags, or waypoints all fail.
"""
from __future__ import annotations

import chess
import pytest

from chessckers_engine.variant_py import moves_black as mb
from chessckers_engine.variant_py.state import parse_fen

cpp = pytest.importorskip("chessckers_cpp")

DIRS = [(-1, -1), (1, -1), (-1, 1), (1, 1)]

# Positions chosen to exercise every branch under the exhaustive sweep:
#   SEED/DENSE  — black king towers among White pawns (normal landings, rams)
#   LINE        — a long diagonal of Whites (multi-capture, stacked rams)
#   OVERSHOOT   — White on h8 from g7: rim landing + off-grid overshoot (keep both)
#   FRIENDLY    — a friendly Black tower mid-diagonal (trace blocked, no overshoot)
#   RIMLAND     — captures then a rim-T landing near the board edge
CORPUS = [
    "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
    "8/8/3kkk2/2PPPP2/2PPPP2/8/8/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
    "k7/1N6/2N5/3N4/4N3/5N2/8/4K3[a8:kkkkk] b - - 0 1",
    "7Q/6k1/8/8/8/8/8/4K3[g7:kkk] b - - 0 1",
    "8/8/8/3k4/2k5/1P6/8/4K3[d5:kkk,c4:kk] b - - 0 1",
    "8/8/8/8/8/8/1P4k1/P6P[g2:kkk] b - - 0 1",
    "7p/8/8/8/8/8/8/4K3[h8:ssss] b - - 0 1",
    "8/8/3kkk2/8/8/8/1PPPPPP1/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
]


def _hop_py(h: mb.CaptureHop) -> tuple:
    return (
        tuple(h.direction),
        h.landing_key,
        h.landing_square,
        tuple(h.captures),
        tuple(h.waypoints),
        h.is_suicide,
        h.crossed_rank1,
        h.cadence,
        h.is_overshoot,
    )


def _hop_cpp(d: dict) -> tuple:
    return (
        tuple(d["direction"]),
        d["landing_key"],
        d["landing_square"],
        tuple(d["captures"]),
        tuple(d["waypoints"]),
        d["is_suicide"],
        d["crossed_rank1"],
        d["cadence"],
        d["is_overshoot"],
    )


@pytest.mark.parametrize("fen", CORPUS)
def test_find_capture_hops_matches_pyvariant(fen: str):
    st = parse_fen(fen)
    board = st.board
    occ = int(board.occupied)
    occw = int(board.occupied_co[chess.WHITE])
    stacks = dict(st.stacks)
    stacks_cpp = {int(sq): pieces for sq, pieces in stacks.items()}

    total_hops = 0
    for r0 in range(8):
        for f0 in range(8):
            for df0, dr0 in DIRS:
                for n in range(1, 9):
                    py_hops = [_hop_py(h) for h in mb._find_capture_hops(board, f0, r0, df0, dr0, n, stacks)]
                    cpp_hops = [_hop_cpp(d) for d in cpp.find_capture_hops(occ, occw, stacks_cpp, f0, r0, df0, dr0, n)]
                    assert cpp_hops == py_hops, (
                        f"mismatch f0={f0} r0={r0} dir=({df0},{dr0}) n={n}\n"
                        f"  fen={fen}\n  cpp={cpp_hops}\n   py={py_hops}"
                    )
                    total_hops += len(py_hops)
    assert total_hops > 0, f"corpus position produced no hops at all: {fen}"


def test_overshoot_and_rim_landing_both_emitted():
    """The g7->h8 White case must yield BOTH a rim-T landing (cadence 2) and a
    distinct off-grid overshoot (cadence 3) at key 'i9' — the keep-both case."""
    st = parse_fen("7Q/6k1/8/8/8/8/8/4K3[g7:kkk] b - - 0 1")
    occ = int(st.board.occupied)
    occw = int(st.board.occupied_co[chess.WHITE])
    stacks_cpp = {int(s): p for s, p in st.stacks.items()}
    hops = cpp.find_capture_hops(occ, occw, stacks_cpp, 6, 6, 1, 1, 3)  # g7 -> (1,1), height 3
    keyed = [(h["landing_key"], h["cadence"], h["is_overshoot"]) for h in hops]
    assert ("i9", 2, False) in keyed
    assert ("i9", 3, True) in keyed
