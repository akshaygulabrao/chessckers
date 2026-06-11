"""The Chessckers training-record chunk codec (training_chunk.py) — the data-only
gzipped-JSON format that replaced pickled games (Phase C of the lc0 wire alignment).

Asserts:
  - round-trip fidelity of every AZExample field, incl. realistic legal-move dicts
    (waypoints / deployCount / demotionsRequired / promotion / capture),
  - the trainer reconstructs BYTE-IDENTICAL encode_position/encode_move tensors from
    a decoded chunk (so the format swap can't silently relabel training data),
  - data-only: a pickle (the OLD format) or garbage decodes to ChunkDecodeError —
    it is never unpickled/executed,
  - the payload is gzip (magic bytes) and schema-tagged.
"""
from __future__ import annotations

import gzip
import json
import pickle

import pytest
import torch

from chessckers_engine.encoding import encode_move, encode_position
from chessckers_engine.selfplay_az import AZExample
from chessckers_engine.training_chunk import (
    SCHEMA,
    ChunkDecodeError,
    decode_chunk,
    encode_chunk,
)


def _game() -> list[AZExample]:
    """Two positions exercising the move-dict shapes encode_move actually reads: a
    plain chess move and a Black diagonal-capture chain with rim waypoints + an
    orthogonal deploy (capture / waypoints / deployCount / demotionsRequired /
    promotion / None values)."""
    return [
        AZExample(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            legal_moves=[
                {"uci": "e2e4", "from": "e2", "to": "e4", "piece": "P"},
                {"uci": "g1f3", "from": "g1", "to": "f3", "piece": "N"},
            ],
            visit_distribution=[0.7, 0.3],
            wdl_target=[1.0, 0.0, 0.0],
            search_wdl=[0.8, 0.15, 0.05],   # Lever 3: search value present on this example
            moves_left_target=12.0,
        ),
        AZExample(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            legal_moves=[
                {"uci": "c3a1", "from": "c3", "to": "a1", "piece": "k",
                 "capture": "b2", "waypoints": ["b2", "z0"],
                 "deployCount": None, "demotionsRequired": None, "promotion": None},
                {"uci": "d6d5", "from": "d6", "to": "d5", "piece": "s",
                 "deployCount": 2, "demotionsRequired": 1, "promotion": "q"},
            ],
            visit_distribution=[0.4, 0.6],
            wdl_target=[0.0, 0.0, 1.0],
            moves_left_target=3.0,
        ),
    ]


def test_roundtrip_preserves_every_field():
    game = _game()
    back = decode_chunk(encode_chunk(game))
    assert len(back) == len(game)
    for a, b in zip(game, back):
        assert b.fen == a.fen
        assert b.legal_moves == a.legal_moves   # move dicts verbatim (incl. None values)
        assert b.visit_distribution == a.visit_distribution
        assert b.wdl_target == a.wdl_target
        assert b.search_wdl == a.search_wdl     # incl. None on the example with no search value
        assert b.moves_left_target == pytest.approx(a.moves_left_target)


def test_decoded_chunk_yields_identical_training_tensors():
    """The whole point of the swap being safe: the trainer must encode a decoded
    example to the SAME tensors as the original — through the real (Rust or Python)
    encoder — or it would relabel positions. This is the end-to-end guarantee."""
    game = _game()
    back = decode_chunk(encode_chunk(game))
    for a, b in zip(game, back):
        assert torch.equal(encode_position(a.fen), encode_position(b.fen))
        for ma, mb in zip(a.legal_moves, b.legal_moves):
            assert torch.equal(encode_move(ma), encode_move(mb))


def test_payload_is_gzip_and_schema_tagged():
    blob = encode_chunk(_game())
    assert blob[:2] == b"\x1f\x8b"                       # gzip magic
    payload = json.loads(gzip.decompress(blob))
    assert payload["schema"] == SCHEMA


def test_decode_rejects_pickle_and_garbage():
    """Data-only: a real pickle (the OLD on-wire format) or arbitrary bytes must
    raise ChunkDecodeError — never unpickle/execute. ChunkDecodeError is a
    ValueError, so the drains' `except (OSError, ChunkDecodeError)` skip the file."""
    poisoned = pickle.dumps(_game(), protocol=pickle.HIGHEST_PROTOCOL)
    with pytest.raises(ChunkDecodeError):
        decode_chunk(poisoned)
    with pytest.raises(ChunkDecodeError):
        decode_chunk(b"not gzip at all")
    wrong_schema = gzip.compress(json.dumps({"schema": "nope", "examples": []}).encode())
    with pytest.raises(ChunkDecodeError):
        decode_chunk(wrong_schema)
    assert isinstance(ChunkDecodeError(), ValueError)


def test_empty_game_roundtrips():
    assert decode_chunk(encode_chunk([])) == []
