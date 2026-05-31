"""Stage A endgame tablebase tests (Phase 1: White lone King vs Black <= N).

Default run is FAST (<~60s): the heavy paths — full N=2 enumeration round-trip
and the N<=1 driver-vs-reference solve — are marked `slow`. The fast core
covers the index/store/mirror invariants and the curriculum seed.

Run with the MAIN checkout's venv (it has the compiled Rust extension), pointing
PYTHONPATH at THIS worktree:

    cd <worktree>/engine && PYTHONPATH=$(pwd) \
      /Users/ox/AAworkspace/chessckers/engine/.venv/bin/python -m pytest \
      tests/test_tablebase_stageA.py -v
"""
from __future__ import annotations

import random

import pytest

import endgame_solver
import tablebase
from chessckers_engine.variant_py import PyVariantClient
from tb import driver, probe, store
from tb.enumerate import canonical_fen, enumerate_level, mirror_fen
from tb.index import class_size, decode, encode
from tb.model import MaterialClass, black_total, side_to_move

_client = PyVariantClient()


# --------------------------------------------------------------------------- #
# 1. Index round-trip
# --------------------------------------------------------------------------- #

def _assert_roundtrip(fens, total: int) -> None:
    n = class_size(total)
    seen: dict[int, str] = {}
    for f in fens:
        mc, idx = encode(f)
        assert mc == MaterialClass(total)
        assert 0 <= idx < n, f"index {idx} out of [0,{n}) for {f!r}"
        cf = canonical_fen(f)
        assert decode(mc, idx) == cf, f"round-trip failed for {f!r}"
        # No two distinct canonical FENs share an index.
        if idx in seen:
            assert seen[idx] == cf, (
                f"index collision {idx}: {seen[idx]!r} vs {cf!r}"
            )
        else:
            seen[idx] = cf


@pytest.mark.parametrize("total", [0, 1])
def test_index_roundtrip_small(total: int):
    _assert_roundtrip(enumerate_level(total), total)


@pytest.mark.slow
def test_index_roundtrip_n2_sample():
    # enumerate_level(2) is the slow part (~2.3M FENs); sample ~5000 to check.
    fens = enumerate_level(2)
    rng = random.Random(0)
    sample = rng.sample(sorted(fens), min(5000, len(fens)))
    _assert_roundtrip(sample, 2)


# --------------------------------------------------------------------------- #
# 2. Mirror automorphism
# --------------------------------------------------------------------------- #

def _sample(total: int, k: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    fens = sorted(enumerate_level(total))
    return rng.sample(fens, min(k, len(fens)))


def test_mirror_involution_and_invariance_n1():
    for f in _sample(1, 200, 1):
        # mirror is an involution: double-mirror returns to the original
        # (up to the canonical representative).
        assert canonical_fen(mirror_fen(mirror_fen(f))) == canonical_fen(f)
        # canonical_fen is mirror-invariant
        assert canonical_fen(mirror_fen(f)) == canonical_fen(f)


def test_mirror_is_oracle_symmetry_n1():
    # Black-to-move forced-mate distance is invariant under the file mirror.
    for f in _sample(1, 60, 2):
        assert tablebase._oracle_dtm(f, 10) == tablebase._oracle_dtm(
            mirror_fen(f), 10
        )


@pytest.mark.slow
def test_mirror_involution_and_invariance_n2():
    for f in _sample(2, 200, 3):
        assert canonical_fen(mirror_fen(mirror_fen(f))) == canonical_fen(f)
        assert canonical_fen(mirror_fen(f)) == canonical_fen(f)


# --------------------------------------------------------------------------- #
# 3. Store codec
# --------------------------------------------------------------------------- #

def test_byte_codec_roundtrip():
    values = [None, (0, None), (1, 0), (-1, 0)]
    for d in range(0, 64):
        values.append((1, d))
        values.append((-1, d))
    for v in values:
        assert store.byte_decode(store.byte_encode(v)) == v


def test_byte_encode_none_is_void():
    assert store.byte_encode(None) == store.VOID


@pytest.mark.parametrize("bad", [0x01, 0xC0, 0x3F, 0x02, 0xFE])
def test_byte_decode_rejects_unemitted_bytes(bad: int):
    # byte_encode never sets both WIN+LOSS (0xC0) nor a non-zero dtm without a
    # WDL flag (0x01..0x3F); byte_decode must guard against corruption.
    with pytest.raises(ValueError):
        store.byte_decode(bad)


def test_byte_encode_rejects_dtm_overflow():
    with pytest.raises(ValueError):
        store.byte_encode((1, 64))


def test_create_open_class_roundtrip(tmp_path):
    store.create_class(tmp_path, 0)
    arr = store.open_class(tmp_path, 0, "r")
    # Fresh class is all-VOID.
    assert all(int(b) == store.VOID for b in arr)
    del arr
    arr = store.open_class(tmp_path, 0, "r+")
    arr[0] = store.byte_encode((1, 7))
    arr.flush()
    del arr
    arr = store.open_class(tmp_path, 0, "r")
    assert store.byte_decode(int(arr[0])) == (1, 7)


# --------------------------------------------------------------------------- #
# 4. Driver == reference (N<=1)  +  5. dtm minimality (play-out)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_driver_matches_reference_and_dtm_is_minimal(tmp_path):
    driver.generate(tmp_path, 1, workers=2, log=lambda *a: None)

    ref_tables = tablebase.solve_up_to(1)
    reference: dict[str, object] = {}
    for tab in ref_tables.values():
        reference.update(tab)

    wins: list[str] = []
    draws = 0
    for total in (0, 1):
        arr = store.open_class(tmp_path, total, "r")
        mc = MaterialClass(total)
        for idx in range(class_size(total)):
            fen = decode(mc, idx)
            byte = int(arr[idx])
            if fen is None:
                assert byte == store.VOID, f"live byte {byte:#x} in VOID slot {idx}"
                continue
            val = store.byte_decode(byte)
            assert val == reference[fen], (
                f"driver {val} != reference {reference[fen]} for {fen!r}"
            )
            if val[0] == 0:
                draws += 1
            elif val[0] > 0:
                wins.append(fen)
        del arr

    assert draws == 0, "N<=1 has no draws (lone WK vs >=1 tower is always decisive)"
    assert wins, "expected some WIN positions at N<=1"

    # dtm minimality: following best_move for BOTH sides reaches a terminal won
    # by the original side-to-move in EXACTLY probe.dtm plies.
    rng = random.Random(0)
    for fen in rng.sample(wins, min(8, len(wins))):
        val = probe.probe(tmp_path, fen)
        want_dtm = val[1]
        winner_side = side_to_move(fen)  # side-to-move wins (wdl == +1)
        cur = fen
        plies = 0
        while True:
            g = _client.new_game(cur)
            if g.get("status") is not None:
                assert g.get("winner") == winner_side, (
                    f"{fen!r}: terminal won by {g.get('winner')}, "
                    f"expected {winner_side}"
                )
                assert plies == want_dtm, (
                    f"{fen!r}: reached mate in {plies}, probe dtm {want_dtm}"
                )
                break
            mv = probe.best_move(tmp_path, cur)
            assert mv is not None, f"no best_move for {cur!r}"
            cur = _client.make_move(cur, mv)["fen"]
            plies += 1
            assert plies <= 2 * want_dtm + 4, f"play-out runaway from {fen!r}"


# --------------------------------------------------------------------------- #
# 6. Curriculum seed (single-king N=2 forced mate via the oracle)
# --------------------------------------------------------------------------- #

def test_curriculum_seed_is_n2_forced_black_mate():
    # The d5/e5 4-King seed is N=4 (infeasible to solve here). Use an N=2
    # Black-to-move forced mate instead: two lone Black Kings adjacent to the
    # White King, Black to move.
    seed = "8/8/8/8/8/8/3kk3/4K3[d2:k,e2:k] b - - 0 1"
    assert black_total(seed) == 2
    # parses cleanly
    g = _client.new_game(seed)
    assert g.get("status") is None
    dtm = endgame_solver.distance_to_mate(seed, 16)
    assert dtm is not None and dtm >= 1, f"expected finite Black-mate dtm, got {dtm}"
