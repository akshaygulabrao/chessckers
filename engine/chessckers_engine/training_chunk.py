"""Chessckers self-play training chunk — the on-disk + on-wire record format for
one game's AZExamples. Replaces the pickled ``list[AZExample]`` (Phase C of the
lc0 wire alignment).

WHY data-only (gzipped JSON, never pickle): a self-play CLIENT — including an
untrusted volunteer / LAN box — uploads these to the trainer, which then reads
them. ``pickle.load`` on attacker-controlled bytes is arbitrary code execution;
``json.loads(gzip.decompress(...))`` is not. gzip because the per-position
legal-move dicts are highly repetitive and compress ~10x, so a game is SMALLER on
the wire than the old uncompressed pickle.

Schema (the gunzipped bytes are UTF-8 JSON):

    {"schema": "ccz1",
     "examples": [{"fen": str,
                   "legal_moves": [ <PyVariantClient move dict>, ... ],
                   "visit_distribution": [float, ...],   # aligned with legal_moves
                   "wdl_target": [float, float, float],
                   "search_wdl": [float, float, float] or null,  # search root value (Lever 3); optional
                   "moves_left_target": float}, ...]}

one entry per AZExample, faithfully. ``legal_moves`` are the move dicts verbatim
(already JSON-shaped); ``encode_move`` reads from/to/capture/waypoints/deployCount/
demotionsRequired/promotion off them, so the trainer reconstructs byte-identical
position/move tensors from a decoded chunk.

The file keeps the historical ``.pkl`` name — the buffer/upload/validation glue
globs ``*.pkl`` in ~10 places; only the BYTES change. Content is self-identifying
by the gzip magic, so a stray real pickle decodes to ChunkDecodeError and is
skipped, not mis-read. (The trainer's separate cold-tier archive, ``games-*.pkl``,
is still pickle — it only ever reads its own writes, never untrusted input.)
"""
from __future__ import annotations

import gzip
import json
import zlib

from chessckers_engine.selfplay_az import AZExample

SCHEMA = "ccz1"


class ChunkDecodeError(ValueError):
    """Bytes are not a valid ccz chunk (bad gzip / JSON / schema / field). Subclasses
    ValueError so existing drain loops that ``except ValueError`` still skip+retry."""


def encode_chunk(examples: list[AZExample]) -> bytes:
    """Serialize one game's AZExamples to a gzipped-JSON chunk (see module doc)."""
    payload = {
        "schema": SCHEMA,
        "examples": [
            {
                "fen": e.fen,
                "legal_moves": e.legal_moves,
                "visit_distribution": e.visit_distribution,
                "wdl_target": e.wdl_target,
                "search_wdl": e.search_wdl,
                "moves_left_target": e.moves_left_target,
            }
            for e in examples
        ],
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    # mtime=0 -> deterministic output (no wall-clock leaks into the payload).
    return gzip.compress(raw, compresslevel=6, mtime=0)


def decode_chunk(data: bytes) -> list[AZExample]:
    """Inverse of encode_chunk. Raises ChunkDecodeError on any malformed / foreign
    payload (truncated gzip, bad JSON, wrong-or-absent schema, missing field) so
    callers skip the file and retry — NEVER executes code carried in ``data``."""
    try:
        payload = json.loads(gzip.decompress(data))
    except (OSError, EOFError, zlib.error, ValueError) as e:
        # BadGzipFile (OSError) / torn gzip (EOFError) / zlib err / JSON (ValueError)
        raise ChunkDecodeError(f"undecodable chunk: {e}") from e
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        got = payload.get("schema") if isinstance(payload, dict) else type(payload).__name__
        raise ChunkDecodeError(f"not a {SCHEMA} chunk (schema={got!r})")
    try:
        return [
            AZExample(
                fen=x["fen"],
                legal_moves=x["legal_moves"],
                visit_distribution=x["visit_distribution"],
                wdl_target=x["wdl_target"],
                moves_left_target=x["moves_left_target"],
                search_wdl=x.get("search_wdl"),  # optional: absent in pre-Lever-3 / scalar-only chunks
            )
            for x in payload["examples"]
        ]
    except (KeyError, TypeError) as e:
        raise ChunkDecodeError(f"chunk missing field: {e}") from e
