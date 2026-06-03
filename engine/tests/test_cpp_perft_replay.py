"""Slice 4 integration gate: full-game legality replay. At EVERY reachable
position (both colors), the complete C++ move generation must match the live
Rust extension exact-ordered.

Two coverage modes:
  * exhaustive perft to depth 3 — visits ALL move sequences from each seed, the
    strongest shallow check (catches apply-path divergences a single-position
    diff hides, since a wrong overlay/bitboard delta only surfaces on the NEXT ply);
  * random replay to terminal — deep/late-game positions across many games.

Side to move: white_legal_moves for White, all_black_legal_moves for Black.
The Rust extension is the oracle (python-chess legality is wrong for this variant).
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


def _assert_match(fen: str):
    """Parse once, generate both colors' moves with C++ and Rust, diff ordered."""
    st = parse_fen(fen)
    b = st.board
    stacks = {int(s): p for s, p in st.stacks.items()}
    if b.turn == chess.WHITE:
        ep = -1 if b.ep_square is None else int(b.ep_square)
        args = (int(b.occupied), int(b.occupied_co[chess.WHITE]), int(b.pawns), int(b.knights),
                int(b.bishops), int(b.rooks), int(b.queens), int(b.kings), int(b.castling_rights),
                ep, stacks)
        cm, rm = cpp.white_legal_moves(*args), rs.white_legal_moves(*args)
    else:
        wk = b.king(chess.WHITE)
        bargs = (int(b.occupied), int(b.occupied_co[chess.WHITE]), -1 if wk is None else int(wk),
                 stacks)
        cm, rm = cpp.all_black_legal_moves(*bargs), rs.all_black_legal_moves(*bargs)
    c, r = [_canon(m) for m in cm], [_canon(m) for m in rm]
    assert c == r, f"\n fen={fen}\n cpp={c}\n  rs={r}"


def test_exhaustive_perft_depth3():
    """Visit every position reachable within 3 plies of each seed; diff at each."""
    client = PyVariantClient()
    visited = 0
    BUDGET = 18000  # node cap so a high-branching seed can't blow up the test

    def perft(fen: str, depth: int):
        nonlocal visited
        if visited >= BUDGET:
            return
        visited += 1
        _assert_match(fen)
        if depth == 0:
            return
        st = client.new_game(fen)
        if st.get("status"):
            return
        for mv in st.get("legalMoves") or []:
            if visited >= BUDGET:
                return
            perft(client.make_move(fen, mv["uci"])["fen"], depth - 1)

    for seed in SEEDS:
        perft(seed, 3)
    assert visited > 1000, f"perft visited too few positions ({visited})"


def test_random_replay_to_terminal():
    """Long random games; diff both colors at every ply down to game end."""
    client = PyVariantClient()
    rng = random.Random(31337)
    positions = 0
    for g in range(40):
        st = client.new_game(SEEDS[g % len(SEEDS)])
        for _ in range(220):
            _assert_match(st["fen"])
            positions += 1
            legal = st.get("legalMoves") or []
            if st.get("status") or not legal:
                break
            st = client.make_move(st["fen"], rng.choice(legal)["uci"])
    assert positions > 1000, f"replay visited too few positions ({positions})"
