"""Slice 2b oracle test: C++ Black quiet diagonals (+ back-rank sprint) and
deploys vs the pure-Python references (moves_black.black_diagonal_quiet_moves /
black_deploy_moves) as canonical sets.

These two generators are not individually exposed by the Rust extension, so the
exact-order-vs-Rust check is deferred to Slice 2d (all_black_legal_moves, which
assembles quiets + deploys + charges + chains in the authoritative order).
"""
from __future__ import annotations

import chess
import pytest

from chessckers_engine.variant_py import moves_black as mb
from chessckers_engine.variant_py.state import STARTING_FEN, parse_fen

cpp = pytest.importorskip("chessckers_cpp")

# Black-to-move full board (STARTING_FEN is White-to-move; the Python public
# generators early-return [] when it isn't Black's turn, but the bb-level C++/Rust
# functions are turn-agnostic — so the comparison must use Black-to-move FENs).
START_BLACK = STARTING_FEN.replace(" w ", " b ")

CORPUS = [
    START_BLACK,                                                         # sprints + friendly merges
    "8/8/3kkk2/8/8/8/8/4K3[d6:kkkk,e6:sk,f6:ssk] b - - 0 1",            # tall towers -> deploys
    "7p/8/8/8/8/8/8/4K3[h8:ssss] b - - 0 1",                           # stone tower deploys/quiets
    "p6p/8/8/8/8/8/8/4K3[a8:s,h8:s] b - - 0 1",                        # corner sprints
    "8/8/8/3k4/2k5/8/8/4K3[d5:kk,c4:k] b - - 0 1",                     # friendly merge (emit + stop)
    "8/8/3kkk2/8/8/8/8/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",              # height-2: quiets + deploys
]


def _bb(state):
    return int(state.board.occupied), int(state.board.occupied_co[chess.WHITE]), {
        int(s): p for s, p in state.stacks.items()
    }


def _canon(m: dict) -> tuple:
    return (m["uci"], m["from"], m["to"], m["piece"], m.get("color"), m.get("deployCount"))


@pytest.mark.parametrize("fen", CORPUS)
def test_quiet_moves_setdiff_vs_python(fen: str):
    state = parse_fen(fen)
    occ, occw, stacks = _bb(state)
    py_set = {_canon(m) for m in mb.black_diagonal_quiet_moves(state)}
    cpp_set = {_canon(m) for m in cpp.black_diagonal_quiet_moves(occ, occw, stacks)}
    assert cpp_set == py_set, (
        f"\n only in cpp: {sorted(cpp_set - py_set)}\n only in py:  {sorted(py_set - cpp_set)}"
    )


@pytest.mark.parametrize("fen", CORPUS)
def test_deploy_moves_setdiff_vs_python(fen: str):
    state = parse_fen(fen)
    occ, occw, stacks = _bb(state)
    py_set = {_canon(m) for m in mb.black_deploy_moves(state)}
    cpp_set = {_canon(m) for m in cpp.black_deploy_moves(occ, occw, stacks)}
    assert cpp_set == py_set, (
        f"\n only in cpp: {sorted(cpp_set - py_set)}\n only in py:  {sorted(py_set - cpp_set)}"
    )


def test_corpus_exercises_quiets_and_deploys():
    # the tall-tower position must produce both quiets and deploys
    state = parse_fen(CORPUS[1])
    occ, occw, stacks = _bb(state)
    assert cpp.black_diagonal_quiet_moves(occ, occw, stacks)
    assert cpp.black_deploy_moves(occ, occw, stacks)
