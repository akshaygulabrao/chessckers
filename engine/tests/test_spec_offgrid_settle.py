"""Spec-derived tests for §3B off-grid overshoot + new capture notation.

Ground truth is chessckers.md §3B (the worked examples), NOT either engine.
These exercise the pure-Python move-gen (the Rust mirror lags the §3B rules,
so we bypass it here) and pin the intended behavior + notation:

    c<N>:<from>~<hops>->rest   (cadence leading; rest always shown and always
                                on-board; cadence distinguishes a rim landing
                                from an off-grid overshoot — no '*' marker)
"""
from __future__ import annotations

from chessckers_engine.variant_py import PyVariantClient

# g3 example: King tucked on a1, so the Rook on h2 is the only target.
G3_FEN = "8/8/8/8/8/6k1/7R/K7[g3:sk] b - - 0 1"
# f5 example: cadence-3 capture of the Rook (h3, d=2), chain on to the Knight (h1).
F5_FEN = "8/8/8/5k2/8/7R/8/4K2N[f5:sk] b - - 0 1"


def _ucis(fen: str) -> set[str]:
    return {m["uci"] for m in PyVariantClient().new_game(fen)["legalMoves"]}


def test_g3_two_distinct_rook_captures():
    ucis = _ucis(G3_FEN)
    # k=2: jump the Rook, land on rim i1, fall back to h2.
    assert "c2:g3~i1->h2" in ucis
    # k=3: overshoot past i1, settle back on h2 — distinct candidate, told apart by cadence.
    assert "c3:g3~i1->h2" in ucis


def test_f5_offgrid_overshoot_chain():
    # Full chain: Rook (h3) then Knight (h1), settling on h1 via an off-grid
    # overshoot past g0. This is the off-grid-overshoot feature.
    assert "c3:f5~i2~g0->h1" in _ucis(F5_FEN)


def test_f5_optional_early_stop():
    # Stopping after the first hop (Rook only) is legal: dead-end on rim i2 -> fall back to h3.
    # Both this and the full chain to h1 must be present (continuing is optional, §3B).
    ucis = _ucis(F5_FEN)
    assert "c3:f5~i2->h3" in ucis        # stop after the Rook
    assert "c3:f5~i2~g0->h1" in ucis     # continue to the Knight
