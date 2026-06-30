#!/usr/bin/env python3
"""Detect oracle-illegal moves in recorded self-play chunks.

The production engine (the akshay-chessckers-0 lc0 fork) plays the games; PyVariant
is the rules authority. If the fork's rules diverge from PyVariant, the fork records
positions whose ply-to-ply transition NO PyVariant-legal move can reproduce — i.e. the
fork played a move the oracle calls illegal. This is exactly the class of bug that the
ply-75 quiet-diagonal-through-a-White-pawn divergence was (training.218.gz).

We replay each chunk's stored positions: for every consecutive pair (exs[i] -> exs[i+1]),
we ask whether SOME move the engine listed as legal at exs[i] actually produces exs[i+1]
when applied through PyVariant. If none does, the played move is oracle-illegal and we
flag it. (Same check watch_game._moves_from_chunk uses to emit '?'.)

Usage:
  check_chunk_parity.py CHUNK.gz [CHUNK.gz ...]
Exit code: 0 = all transitions oracle-legal; 1 = at least one illegal transition found.
"""
import sys

from chessckers_engine.training_chunk import decode_chunk
from chessckers_engine.variant_py import PyVariantClient


def _norm(fen: str) -> str:
    """Canonicalize a FEN for transition comparison by blanking the en-passant
    field. En passant is disabled in this variant (board.cc kPawnMask=0, no ep
    capture), but PyVariant is built on python-chess, which still records an ep
    target square (e.g. g3) after a White pawn double-step while the fork records
    '-'. That field is semantically dead here, so comparing it would spuriously
    flag every White double-step. Board/overlay/turn/castling/clocks stay strict —
    those are where a real illegal-move divergence (e.g. the ply-75 bug) shows up."""
    head, _, rest = fen.partition(" ")  # head = '<board>[<overlay>]'
    parts = rest.split()
    if len(parts) >= 3:  # [turn, castling, ep, halfmove, fullmove, {ckstate}?]
        parts[2] = "-"
    return head + " " + " ".join(parts)


def check_chunk(path: str, client: PyVariantClient) -> list[dict]:
    """Return a list of illegal-transition records for one chunk (empty = clean).

    Authoritative check: does ANY *PyVariant*-legal move from `before` reproduce
    `after`? We generate PyVariant's own legal moves rather than trusting the chunk's
    stored (fork-generated) move list — the question is reachability under the oracle,
    independent of how the fork notated or enumerated its moves."""
    exs = decode_chunk(open(path, "rb").read())
    bad: list[dict] = []
    for i in range(len(exs) - 1):
        before, after = exs[i].fen, exs[i + 1].fen
        after_norm = _norm(after)
        try:
            legal = client.new_game(before)["legalMoves"]
        except Exception as e:  # noqa: BLE001 — an unparseable position is itself a divergence
            bad.append({"ply": i + 1, "before": before, "after": after, "note": f"unparseable: {e}"})
            continue
        connected = False
        for m in legal:
            try:
                if _norm(client.make_move(before, m["uci"])["fen"]) == after_norm:
                    connected = True
                    break
            except Exception:  # noqa: BLE001 — defensive; PyVariant-legal moves should apply
                continue
        if not connected:
            bad.append({"ply": i + 1, "before": before, "after": after})
    return bad


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        print("usage: check_chunk_parity.py CHUNK.gz [CHUNK.gz ...]")
        return 2
    client = PyVariantClient()
    total_bad = 0
    for path in paths:
        try:
            bad = check_chunk(path, client)
        except Exception as e:  # noqa: BLE001 — undecodable/corrupt chunk is itself a flag
            print(f"!! {path}: could not decode ({e})")
            total_bad += 1
            continue
        if bad:
            total_bad += len(bad)
            name = path.rsplit("/", 1)[-1]
            for b in bad:
                print(f"!! ILLEGAL {name} ply {b['ply']}: no PyVariant-legal move connects")
                print(f"     before: {b['before']}")
                print(f"     after : {b['after']}")
    n = len(paths)
    if total_bad:
        print(f"\nFAIL: {total_bad} illegal transition(s) across {n} chunk(s).")
        return 1
    print(f"OK: {n} chunk(s) scanned, all transitions oracle-legal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
