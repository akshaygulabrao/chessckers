"""Slice 3a oracle test: the C++ Chessckers check predicate vs the live Rust
extension.

  * black_can_capture_white_king  — full diagonal chain/ram search to the king
  * square_attacked_by_black_chessckers — walk-based attack on a target square
  * white_in_chessckers_check      — the OR of the two on the king square

Positions come from random self-play rollouts (real, diverse king-attack states)
plus hand-crafted in-check positions. square_attacked is swept over all 64
target squares on a subset. python-chess's is_check is NOT a valid oracle here
(it mis-models the Black-King encoding), so the Rust extension is the oracle.
"""
from __future__ import annotations

import random

import chess
import pytest

from chessckers_engine.variant_py.client import PyVariantClient
from chessckers_engine.variant_py.state import STARTING_FEN, parse_fen

cpp = pytest.importorskip("chessckers_cpp")
rs = pytest.importorskip("chessckers_movegen")

SEEDS = [
    "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
    "8/8/3kkk2/2PPPP2/2PPPP2/8/8/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
    "7p/8/8/8/8/8/8/4K3[h8:ssss] b - - 0 1",
    STARTING_FEN,
]

# Hand-crafted positions where Black CAN capture the White king (true branch).
CHECK_TRUE = [
    "8/8/8/8/8/2k5/1K6/8[c3:kk] w - - 0 1",      # diagonal chain c3 -> b2(king)
    "8/8/8/8/8/8/2k5/1K6[c2:kkk] w - - 0 1",     # charge/diagonal onto b-file king
]


def _collect_positions(n_games=10, max_plies=45, seed=20260603):
    client = PyVariantClient()
    rng = random.Random(seed)
    fens = []
    for g in range(n_games):
        st = client.new_game(SEEDS[g % len(SEEDS)])
        for _ in range(max_plies):
            fens.append(st["fen"])
            legal = st.get("legalMoves") or []
            if st.get("status") or not legal:
                break
            mv = rng.choice(legal)
            st = client.make_move(st["fen"], mv["uci"])
    return fens + CHECK_TRUE


def _bb(fen):
    b = parse_fen(fen).board
    occ = int(b.occupied)
    occw = int(b.occupied_co[chess.WHITE])
    stacks = {int(s): p for s, p in parse_fen(fen).stacks.items()}
    wk = b.king(chess.WHITE)
    king_sq = -1 if wk is None else int(wk)
    return occ, occw, stacks, king_sq


def test_check_predicate_matches_rust_over_rollout():
    fens = _collect_positions()
    assert len(fens) > 100, "rollout produced too few positions"
    saw_true = saw_false = False
    for i, fen in enumerate(fens):
        occ, occw, stacks, king_sq = _bb(fen)

        cpp_cap = cpp.black_can_capture_white_king(occ, occw, king_sq, stacks)
        rs_cap = rs.black_can_capture_white_king(occ, occw, king_sq, stacks)
        assert cpp_cap == rs_cap, f"black_can_capture_white_king @ {fen}"
        saw_true |= bool(cpp_cap)
        saw_false |= not cpp_cap

        rs_check = rs_cap or (
            king_sq >= 0 and rs.square_attacked_by_black_chessckers(occ, occw, stacks, king_sq)
        )
        assert cpp.white_in_chessckers_check(occ, occw, king_sq, stacks) == rs_check, (
            f"white_in_chessckers_check @ {fen}"
        )

        # exhaustive 64-square attack sweep on a subset (keeps the test quick)
        if i % 8 == 0:
            for t in range(64):
                assert cpp.square_attacked_by_black_chessckers(
                    occ, occw, stacks, t
                ) == rs.square_attacked_by_black_chessckers(occ, occw, stacks, t), (
                    f"square_attacked @ {fen} target={t}"
                )

    assert saw_true and saw_false, "rollout must include both in-check and safe positions"


@pytest.mark.parametrize("fen", CHECK_TRUE)
def test_handcrafted_check_positions_are_check(fen: str):
    occ, occw, stacks, king_sq = _bb(fen)
    assert cpp.black_can_capture_white_king(occ, occw, king_sq, stacks)
    assert cpp.white_in_chessckers_check(occ, occw, king_sq, stacks)
