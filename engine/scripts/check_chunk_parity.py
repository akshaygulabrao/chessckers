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

PCR gap awareness: the fork emits training records for only ~25% of plies under
playout-cap randomization. Consecutive records may be several plies apart.  When
no single legal move connects a pair we check whether the transition has a
plausible gap signature (side-to-move repeats, raw halfmove jump >1, or ply-delta >1
when the `ply` field is present) before classifying as illegal.  If FEN heuristics
don't fire, a bounded multi-hop reachability probe (up to 3 PyVariant-legal plies)
is used as a final fallback — if the after-position is reachable within 3 plies it is
a PCR gap, not a rules divergence.  Only transitions unreachable within 3 hops AND
with no FEN gap-signature are reported as ILLEGAL.

Usage:
  check_chunk_parity.py CHUNK.gz [CHUNK.gz ...]
Exit code: 0 = all transitions oracle-legal (gaps are OK); 1 = at least one illegal
transition found.
"""
import gzip
import json
import sys

from chessckers_engine.training_chunk import decode_chunk
from chessckers_engine.variant_py import PyVariantClient


def _norm(fen: str) -> str:
    """Canonicalize a FEN for transition comparison by blanking the en-passant
    field and the move counters. En passant is disabled in this variant (board.cc
    kPawnMask=0, no ep capture), but PyVariant is built on python-chess, which
    still records an ep target square (e.g. g3) after a White pawn double-step
    while the fork records '-'. The halfmove/fullmove counters are likewise
    semantically dead for "which move connects position i to i+1" (no 50-move
    rule; fullmove is derivable) and the two writers disagree: the fork's chunk
    FENs freeze fullmove at 1, while PyVariant ticks it on Black moves (fixed
    2026-07-16 for engine temp-decay) — a strict compare would spuriously flag
    every Black move. Board/overlay/turn/castling stay strict — those are where
    a real illegal-move divergence (e.g. the ply-75 bug) shows up."""
    head, _, rest = fen.partition(" ")  # head = '<board>[<overlay>]'
    parts = rest.split()
    if len(parts) >= 5:  # [turn, castling, ep, halfmove, fullmove, {ckstate}?]
        parts[2] = "-"
        parts[3] = "0"
        parts[4] = "1"
    elif len(parts) >= 3:
        parts[2] = "-"
    return head + " " + " ".join(parts)


def _is_gap_signature(before_fen: str, after_fen: str,
                      before_raw: dict | None = None,
                      after_raw: dict | None = None) -> bool:
    """Return True when the (before -> after) FEN transition shows a clear FEN-level
    gap signature (fast, no PyVariant calls needed).

    Detected signals:
      - `ply` field present in BOTH raw records and after_raw["ply"] - before_raw["ply"] > 1
      - side-to-move (FEN turn field) repeats (same side plays again → ≥2 plies elapsed)
      - absolute raw halfmove-clock difference > 1 (more half-moves than 1 ply can produce)

    These are necessary-condition checks: if none fires we fall back to the multi-hop
    reachability probe in check_chunk (which is authoritative but costs PyVariant calls).
    """
    # ply-delta check (future-proofed; field not yet emitted by current fork)
    if (before_raw is not None and after_raw is not None
            and "ply" in before_raw and "ply" in after_raw):
        try:
            if int(after_raw["ply"]) - int(before_raw["ply"]) > 1:
                return True
        except (ValueError, TypeError):
            pass

    # Parse turn and halfmove from the raw FEN (BEFORE _norm blanks them).
    def _fen_parts(fen: str):
        # FEN: <board>[<overlay>] <turn> <castling> <ep> <halfmove> <fullmove> [{ckstate}?]
        _, _, rest = fen.partition(" ")
        return rest.split()

    b_parts = _fen_parts(before_fen)
    a_parts = _fen_parts(after_fen)

    # turn-repeat: same side to move (field index 0 of the tail)
    if len(b_parts) >= 1 and len(a_parts) >= 1:
        if b_parts[0] == a_parts[0]:
            return True

    # halfmove-clock jump > 1 (field index 3 of the tail)
    if len(b_parts) >= 4 and len(a_parts) >= 4:
        try:
            hm_before = int(b_parts[3])
            hm_after = int(a_parts[3])
            if abs(hm_after - hm_before) > 1:
                return True
        except (ValueError, TypeError):
            pass

    return False


def _reachable_in_hops(before: str, after_norm: str, client: PyVariantClient,
                        max_hops: int = 3) -> bool:
    """DFS: return True if after_norm is reachable from before within max_hops
    PyVariant-legal moves.  Only invoked as a last resort (FEN heuristics didn't fire)
    so it is expected to be a rare path.  max_hops=3 bounds the branching-factor
    explosion while covering the PCR-gap cases observed in practice."""
    def _dfs(fen: str, depth: int) -> bool:
        try:
            legal = client.new_game(fen)["legalMoves"]
        except Exception:  # noqa: BLE001
            return False
        for m in legal:
            try:
                result_fen = client.make_move(fen, m["uci"])["fen"]
            except Exception:  # noqa: BLE001
                continue
            if _norm(result_fen) == after_norm:
                return True
            if depth > 1 and _dfs(result_fen, depth - 1):
                return True
        return False

    return _dfs(before, max_hops)


def check_chunk(path: str, client: PyVariantClient) -> tuple[list[dict], int]:
    """Return (bad_transitions, gap_count) for one chunk.

    bad_transitions: list of oracle-illegal transition dicts (empty = clean).
    gap_count: number of transitions classified as PCR gaps (not illegal).

    Authoritative check: does ANY *PyVariant*-legal move from `before` reproduce
    `after`? We generate PyVariant's own legal moves rather than trusting the chunk's
    stored (fork-generated) move list — the question is reachability under the oracle,
    independent of how the fork notated or enumerated its moves.

    When single-move fails: FEN heuristics classify obvious gaps (side-repeat,
    halfmove-jump, ply-delta).  If heuristics don't fire, a bounded multi-hop
    reachability probe (up to 3 hops) is the final arbiter — if after is reachable
    it's a gap, else it's illegal."""
    data = open(path, "rb").read()
    exs = decode_chunk(data)
    raw_exs = json.loads(gzip.decompress(data))["examples"]

    bad: list[dict] = []
    gaps: int = 0
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
            # Fast path: FEN-level gap signature (no extra PyVariant calls)
            if _is_gap_signature(before, after,
                                 before_raw=raw_exs[i],
                                 after_raw=raw_exs[i + 1]):
                gaps += 1
            # Slow path: bounded multi-hop reachability probe.  Only reached when the
            # fast heuristics didn't fire (rare: PCR gaps where halfmove reset mid-gap
            # makes clock jump appear ≤1).  3 hops covers all observed cases in practice.
            elif _reachable_in_hops(before, after_norm, client, max_hops=3):
                gaps += 1
            else:
                bad.append({"ply": i + 1, "before": before, "after": after})
    return bad, gaps


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        print("usage: check_chunk_parity.py CHUNK.gz [CHUNK.gz ...]")
        return 2
    client = PyVariantClient()
    total_bad = 0
    total_gaps = 0
    gap_chunks = 0
    for path in paths:
        try:
            bad, gaps = check_chunk(path, client)
        except Exception as e:  # noqa: BLE001 — undecodable/corrupt chunk is itself a flag
            print(f"!! {path}: could not decode ({e})")
            total_bad += 1
            continue
        if gaps:
            total_gaps += gaps
            gap_chunks += 1
        if bad:
            total_bad += len(bad)
            name = path.rsplit("/", 1)[-1]
            gap_note = f" (gaps={gaps})" if gaps else ""
            for b in bad:
                print(f"!! ILLEGAL {name} ply {b['ply']}: no PyVariant-legal move connects{gap_note}")
                print(f"     before: {b['before']}")
                print(f"     after : {b['after']}")
        elif gaps:
            name = path.rsplit("/", 1)[-1]
            print(f"   {name}: gaps={gaps} (PCR sparse — ok)")
    n = len(paths)
    if total_bad:
        gap_suffix = f" ({total_gaps} gap(s) across {gap_chunks} chunk(s))" if total_gaps else ""
        print(f"\nFAIL: {total_bad} illegal transition(s) across {n} chunk(s).{gap_suffix}")
        return 1
    if total_gaps:
        print(f"OK: {n} chunk(s) scanned, all transitions oracle-legal. "
              f"({total_gaps} gaps across {gap_chunks} chunk(s))")
    else:
        print(f"OK: {n} chunk(s) scanned, all transitions oracle-legal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
