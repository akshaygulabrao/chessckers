"""Resumable, process-parallel tablebase generation driver (Phase 1).

Climbs material classes bottom-up (total Black pieces 0, 1, 2, ...). Each class
is solved by an index/mmap-backed **Jacobi fixpoint** that reuses the reference
solver's win/loss rule (`tablebase.solve_level`) but keeps all state on disk:

  * Init: every live slot is classified; terminal win/loss bytes are written,
    everything else starts 0x00 (draw-or-undetermined — the two are
    indistinguishable by byte and converge to the same answer).
  * Sweep: workers read a committed snapshot of this class plus the final lower
    classes, recompute each still-0x00 position from its successors, and return
    the sparse set of newly-proven (idx, byte). The parent applies them. This is
    Jacobi (no worker sees this sweep's updates), matching the reference.
  * Converge when a sweep proves nothing new; remaining 0x00 = draw.

Successor values: a successor that is terminal is valued inline; otherwise it is
encoded and read from this class (same total) or a final lower class (capture).
Black-captures-White-King successors are terminal, so `encode` is never called
on a kingless position.

`0x00` is read back as the draw value `(0, None)`. During iteration this is
equivalent to the reference treating an undetermined successor as "not yet
proven": wdl 0 neither triggers a win (needs a successor loss) nor permits a
loss (needs all successors winning), so convergence is monotone and identical.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

from tb import store
from tb.index import class_size, decode, encode
from tb.model import MaterialClass, terminal_value

# --------------------------------------------------------------------------- #
# Per-worker process state (built lazily, reused across tasks)
# --------------------------------------------------------------------------- #
_W_CLIENT = None


def _client():
    global _W_CLIENT
    if _W_CLIENT is None:
        from chessckers_engine.variant_py import PyVariantClient

        _W_CLIENT = PyVariantClient()
    return _W_CLIENT


# Edge int64 encoding (so the successor graph fits a compact numpy CSR):
#   reference edge (non-negative): (succ_total << IDX_BITS) | succ_index
#   terminal edge  (negative):     -(value_byte + 1)
IDX_BITS = 40
_IDX_MASK = (1 << IDX_BITS) - 1


# --------------------------------------------------------------------------- #
# Graph build (parallel): movegen + encode happen ONCE per position here, never
# in the sweep loop.
# --------------------------------------------------------------------------- #

def _graph_chunk(args):
    """Classify a slot range. Writes terminal win/loss/draw bytes in place; for
    each non-terminal live position returns its successor edges. Returns
    (idx_array, lengths_array, flat_edges_array) — a per-chunk CSR."""
    import numpy as np

    root, total, lo, hi = args
    cl = _client()
    mm = store.open_class(root, total, "r+")
    mc = MaterialClass(total)
    idxs: list[int] = []
    lengths: list[int] = []
    flat: list[int] = []
    for idx in range(lo, hi):
        fen = decode(mc, idx)
        if fen is None:
            continue  # VOID stays 0xFF
        g = cl.new_game(fen)
        status = g.get("status")
        if status is not None:
            mm[idx] = store.byte_encode(terminal_value(status, g.get("winner"), g["turn"]))
            continue
        mm[idx] = store.DRAW  # undetermined (== draw-for-now)
        edges: list[int] = []
        for m in g["legalMoves"]:
            s2 = cl.make_move(fen, m["uci"])
            st = s2.get("status")
            if st is not None:
                byte = store.byte_encode(terminal_value(st, s2.get("winner"), s2["turn"]))
                edges.append(-(byte + 1))
            else:
                mc2, idx2 = encode(s2["fen"])
                edges.append((mc2.black_total << IDX_BITS) | idx2)
        idxs.append(idx)
        lengths.append(len(edges))
        flat.extend(edges)
    mm.flush()
    del mm
    return (
        np.array(idxs, dtype=np.int64),
        np.array(lengths, dtype=np.int64),
        np.array(flat, dtype=np.int64),
    )


def _chunks(n: int, parts: int) -> list[tuple[int, int]]:
    step = max(1, (n + parts - 1) // parts)
    return [(i, min(i + step, n)) for i in range(0, n, step)]


# --------------------------------------------------------------------------- #
# Class solve: build the graph once (parallel), then Jacobi-sweep over the
# cached edges in-process (pure byte lookups — no movegen/encode).
# --------------------------------------------------------------------------- #

def solve_class(root, total: int, workers: int = 6, log=print) -> dict:
    """Solve one material class into its hd0 class file. Returns solve stats."""
    import numpy as np

    n = class_size(total)
    store.create_class(root, total)
    ranges = _chunks(n, workers * 8)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_graph_chunk, [(root, total, lo, hi) for lo, hi in ranges]))

    idx_arr = np.concatenate([p[0] for p in parts]) if parts else np.empty(0, np.int64)
    lengths = np.concatenate([p[1] for p in parts]) if parts else np.empty(0, np.int64)
    flat = np.concatenate([p[2] for p in parts]) if parts else np.empty(0, np.int64)
    offsets = np.zeros(len(idx_arr) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    log(f"  N{total}: graph built — {len(idx_arr)} non-terminal positions, "
        f"{len(flat)} edges, {len(ranges)} chunks")

    cur = store.open_class(root, total, "r+")
    lowers = {t: store.open_class(root, t, "r") for t in range(total)}

    def resolve(e: int):
        if e < 0:
            return store.byte_decode(-e - 1)
        t = e >> IDX_BITS
        j = e & _IDX_MASK
        return store.byte_decode(int(cur[j] if t == total else lowers[t][j]))

    # Bellman-Ford relaxation: recompute every non-terminal position each sweep
    # and APPLY an update when it is newly decisive OR its dtm improves
    # (decreases). A WIN's dtm must be allowed to shrink as faster loss-children
    # appear; marking a position done at first proof would lock in a non-minimal
    # dtm. Converges when a sweep changes nothing; positions never made decisive
    # stay 0x00 = draw.
    sweep = 0
    while True:
        sweep += 1
        updates: list[tuple[int, int]] = []  # (idx, byte)
        for i in range(len(idx_arr)):
            best_win: int | None = None
            all_wins = True
            saw = False
            for e in flat[offsets[i]:offsets[i + 1]]:
                saw = True
                wdl, dtm = resolve(int(e))
                if wdl < 0:  # successor is a loss for the opponent -> mover wins
                    cand = 1 + (dtm or 0)
                    best_win = cand if best_win is None else min(best_win, cand)
                elif wdl == 0:
                    all_wins = False
            if best_win is not None:
                new = (1, best_win)
            elif all_wins and saw:
                new = (-1, max(1 + (resolve(int(e))[1] or 0)
                               for e in flat[offsets[i]:offsets[i + 1]]))
            else:
                continue
            idx = int(idx_arr[i])
            old = store.byte_decode(int(cur[idx]))  # 0x00 -> (0, None)
            # Apply if newly decisive, or same result with a strictly smaller dtm.
            if old[0] == 0 or (old[0] == new[0] and new[1] < (old[1] or 0)):
                updates.append((idx, store.byte_encode(new)))
        if not updates:
            break
        for idx, byte in updates:
            cur[idx] = byte
        cur.flush()
        log(f"  N{total}: sweep {sweep} updated {len(updates)}")
    del cur

    # Tally (vectorized over the byte histogram).
    final = store.open_class(root, total, "r")
    counts = np.bincount(np.asarray(final), minlength=256)
    del final
    void = int(counts[store.VOID])
    draw = int(counts[store.DRAW])
    win = int(counts[0x40:0x80].sum())   # 0x40..0x7F
    loss = int(counts[0x80:0xC0].sum())  # 0x80..0xBF
    stats = {"win": win, "loss": loss, "draw": draw, "void": void, "sweeps": sweep}
    store.mark_class(root, total, "solved", **stats)
    log(f"  N{total}: solved in {sweep} sweeps  win={win} loss={loss} draw={draw}")
    return stats


def generate(root, max_total: int, workers: int = 6, resume: bool = True, log=print) -> None:
    """Solve classes 0..max_total, lowest first. Skips classes already marked
    solved in the manifest when `resume`."""
    manifest = store.read_manifest(root)
    solved = {
        c["total"] for c in manifest.get("classes", {}).values()
        if c.get("status") == "solved"
    }
    for total in range(max_total + 1):
        if resume and total in solved and store.class_path(root, total).exists():
            log(f"N{total}: already solved (resume)")
            continue
        log(f"=== solving class N{total} (size {class_size(total)}) ===")
        solve_class(root, total, workers=workers, log=log)
