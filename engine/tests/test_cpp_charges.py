"""Slice 2c oracle test: C++ Black charges (orthogonal King-top tower moves with
king-demotion combinatorics + overshoot charges) vs the pure-Python reference
moves_black.black_charge_moves, as canonical sets.

Charges aren't individually exposed by Rust; exact-order-vs-Rust comes at Slice
2d (all_black_legal_moves). Positions exercise demotion choices C(n,d), forced
demotion (n_kings==d), path-capture / ram-with-path-capture, overshoot (rim
fallback), friendly-merge stop, and mixed stone/king towers (resulting top).
"""
from __future__ import annotations

import chess
import pytest

from chessckers_engine.variant_py import moves_black as mb
from chessckers_engine.variant_py.state import parse_fen

cpp = pytest.importorskip("chessckers_cpp")

CORPUS = [
    "8/8/8/8/4k3/8/8/4K3[e4:kkk] b - - 0 1",        # central h=3: C(3,1),C(3,2),forced
    "8/8/8/8/4k3/8/8/4K3[e4:kkkk] b - - 0 1",       # h=4: more demotion choices
    "8/8/8/8/4k3/8/4P3/4K3[e4:kkk] b - - 0 1",      # ram-with-path-capture down to e1
    "8/8/8/4k3/8/4P3/8/4K3[e5:kkkk] b - - 0 1",     # overshoot a White (non-ram charge captures)
    "1k6/8/8/8/8/8/8/4K3[b8:kk] b - - 0 1",         # overshoot charge (rim fallback z8->a8)
    "8/8/8/8/3kk3/8/8/4K3[d4:kk,e4:kk] b - - 0 1",  # friendly merge (emit + stop)
    "8/8/8/8/4k3/8/8/4K3[e4:skk] b - - 0 1",        # mixed tower: resulting-top varies by choice
]


def _bb(state):
    return int(state.board.occupied), int(state.board.occupied_co[chess.WHITE]), {
        int(s): p for s, p in state.stacks.items()
    }


def _canon(m: dict) -> tuple:
    def _t(v):
        return tuple(v) if v is not None else None

    return (
        m["uci"],
        m["from"],
        m["to"],
        m["piece"],
        m.get("color"),
        m.get("capture"),
        _t(m.get("waypoints")),
        _t(m.get("demotedKings")),
        m.get("demotionsRequired"),
        _t(m.get("sourceKingPositions")),
    )


@pytest.mark.parametrize("fen", CORPUS)
def test_charge_moves_setdiff_vs_python(fen: str):
    state = parse_fen(fen)
    occ, occw, stacks = _bb(state)
    py_set = {_canon(m) for m in mb.black_charge_moves(state)}
    cpp_set = {_canon(m) for m in cpp.black_charge_moves(occ, occw, stacks)}
    assert cpp_set == py_set, (
        f"\n only in cpp: {sorted(cpp_set - py_set)}\n only in py:  {sorted(py_set - cpp_set)}"
    )


def test_corpus_exercises_demotions_and_overshoot():
    # demotion choices present in the h=4 position; overshoot waypoint in b8.
    s1 = parse_fen(CORPUS[1])
    assert any(m["demotedKings"] for m in cpp.black_charge_moves(*_bb(s1)))
    s4 = parse_fen(CORPUS[4])
    assert any(m["waypoints"] for m in cpp.black_charge_moves(*_bb(s4)))
