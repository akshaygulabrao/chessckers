"""Stage A Chessckers endgame tablebase generator (correctness-first).

Enumerates a class of positions (White = lone King, Black = total <= N pieces),
solves them exactly bottom-up via a retrograde fixpoint keyed by canonical FEN,
and cross-checks every position against the independent `endgame_solver` oracle.

Value per position is `(wdl, dtm)` from the side-to-move perspective:
  wdl in {+1 win, 0 draw, -1 loss}; dtm = plies to mate (None for draws).

Levels are solved in increasing total-Black-piece order (0, 1, 2, ...). A
position's successors (via `make_move`) hand the move to the opponent and may
drop to a strictly lower level (captures); those are resolved from already-solved
lower tables. Same-level successors evolve during the fixpoint sweeps.

CLI:
    python tablebase.py            # gate: N=2, then N=4
    python tablebase.py --max-n 2
"""
from __future__ import annotations

import argparse
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from chessckers_engine.variant_py import PyVariantClient
from chessckers_engine.variant_py.state import parse_fen, serialize_fen
from endgame_solver import _dtm_black, _dtm_white, _memo
from tb.enumerate import enumerate_level
from tb.model import Value, black_total, side_to_move, terminal_value

_client = PyVariantClient()

TB_DIR = Path(__file__).resolve().parent / "weights" / "tablebase"


# --------------------------------------------------------------------------- #
# Solver (retrograde fixpoint)
# --------------------------------------------------------------------------- #
# Enumeration, the (wdl, dtm) Value type, and the per-FEN helpers
# (`black_total`, `side_to_move`, `terminal_value`) now live in the `tb`
# package — imported above.


def _norm(fen: str) -> str:
    """Clock-normalized serialize key (halfmove 0, fullmove 1), no mirror.

    `enumerate_level` keys are clock-0 serialized FENs, but `make_move`
    successors carry a bumped move clock. Without normalizing, a successor would
    miss the table and be (wrongly) treated as an undetermined draw — which
    silently mislabels forced White wins as draws."""
    st = parse_fen(fen)
    st.board.halfmove_clock = 0
    st.board.fullmove_number = 1
    return serialize_fen(st)


def solve_level(
    total: int, lower: dict[str, Value]
) -> dict[str, Value]:
    """Solve all positions with Black total == `total` given solved `lower`
    tables (all positions with strictly fewer Black pieces). Returns this
    level's table; does NOT include lower-level entries."""
    fens = enumerate_level(total)

    table: dict[str, Value] = {}
    # successors[fen] = list of (succ_fen, succ_total, terminal_value_or_None).
    # A non-None terminal value is used directly (the successor may not be
    # enumerated — e.g. Black capturing the White king leaves a kingless,
    # same-level position that `enumerate_level` never produces).
    successors: dict[str, list[tuple[str, int, Value | None]]] = {}

    # Seed: terminal positions get fixed values; non-terminals start undetermined.
    for fen in fens:
        g = _client.new_game(fen)
        status, winner = g.get("status"), g.get("winner")
        mover = g["turn"]
        if status is not None:
            table[fen] = terminal_value(status, winner, mover)
            continue
        succs: list[tuple[str, int, Value | None]] = []
        for m in g["legalMoves"]:
            s2 = _client.make_move(fen, m["uci"])
            sf = _norm(s2["fen"])
            st = black_total(sf)
            s2_status = s2.get("status")
            if s2_status is not None:
                # Terminal right after the move — value it inline from the
                # successor's side-to-move (the opponent) perspective.
                tv = terminal_value(s2_status, s2.get("winner"), s2["turn"])
            else:
                tv = None
            succs.append((sf, st, tv))
        successors[fen] = succs
        table[fen] = (0, None)  # undetermined placeholder (treated as draw)

    # Determined-set tracks which fens have a settled win/loss value this level.
    determined: dict[str, Value] = {}

    def succ_value(sf: str, st: int, tv: Value | None) -> Value | None:
        """Settled value of a successor, or None if not yet determined."""
        if tv is not None:
            return tv  # terminal successor, valued inline
        if st < total:
            v = lower.get(sf)
            if v is None:
                raise KeyError(f"successor {sf!r} (level {st}) missing from lower table")
            return v
        return determined.get(sf)

    # Fixpoint sweeps over non-terminal, same-level positions.
    pending = set(successors.keys())
    changed = True
    while changed:
        changed = False
        newly: dict[str, Value] = {}
        for fen in list(pending):
            succs = successors[fen]
            # WIN if any successor is a LOSS for the opponent (i.e. wdl == -1).
            best_win_dtm: int | None = None
            all_known_wins = True  # all successors are determined wins-for-opponent
            for sf, st, tv in succs:
                v = succ_value(sf, st, tv)
                if v is None:
                    all_known_wins = False
                    continue
                wdl, dtm = v
                if wdl < 0:  # opponent loses from sf -> mover wins
                    cand = 1 + (dtm or 0)
                    if best_win_dtm is None or cand < best_win_dtm:
                        best_win_dtm = cand
                elif wdl == 0:
                    all_known_wins = False
                # wdl > 0 (opponent wins) contributes to "all wins" check
            if best_win_dtm is not None:
                newly[fen] = (1, best_win_dtm)
                continue
            if all_known_wins and succs:
                # All successors determined and none is a loss-for-opponent;
                # every successor is a win-for-opponent -> mover loses.
                worst = 0
                for sf, st, tv in succs:
                    v = succ_value(sf, st, tv)
                    assert v is not None and v[0] > 0
                    worst = max(worst, 1 + (v[1] or 0))
                newly[fen] = (-1, worst)
        if newly:
            changed = True
            for fen, v in newly.items():
                determined[fen] = v
                pending.discard(fen)

    # Anything still pending after convergence is a draw.
    for fen in pending:
        table[fen] = (0, None)
    for fen, v in determined.items():
        table[fen] = v
    return table


def solve_up_to(max_n: int) -> dict[int, dict[str, Value]]:
    """Solve levels 0..max_n; return {level: table}."""
    tables: dict[int, dict[str, Value]] = {}
    lower: dict[str, Value] = {}
    for total in range(max_n + 1):
        t = solve_level(total, lower)
        tables[total] = t
        lower = {**lower, **t}  # accumulate all solved positions for lookups
    return tables


# --------------------------------------------------------------------------- #
# Cross-check against the oracle
# --------------------------------------------------------------------------- #

def _oracle_dtm(fen: str, depth: int) -> int | None:
    """Oracle's plies-to-forced-Black-mate for the side to move in `fen`."""
    _memo.clear()
    if side_to_move(fen) == "black":
        return _dtm_black(fen, depth)
    return _dtm_white(fen, depth)


def _tb_black_dtm(fen: str, wdl: int, dtm: int | None) -> int | None:
    """TB's 'forced Black mate in d plies' for the side to move, or None
    (draw, or White wins)."""
    mover = side_to_move(fen)
    if wdl > 0 and mover == "black":
        return dtm
    if wdl < 0 and mover == "white":
        return dtm
    return None


def _check_one(args: tuple[str, int | None, int]) -> tuple[str, str, str] | None:
    """Pool worker: compare one position's TB value to the oracle."""
    fen, tb_black_dtm, depth = args
    oracle = _oracle_dtm(fen, depth)
    if tb_black_dtm != oracle:
        return (fen, f"black-mate dtm={tb_black_dtm}", f"oracle dtm={oracle}")
    return None


def _stratified_sample(
    items: list[tuple[str, int, int | None]], k: int, seed: int
) -> list[tuple[str, int, int | None]]:
    """Pick ~k positions biased toward decisive ones. Decisive (Black win or
    White win) positions are where dtm bugs hide; draws dominate by count but
    are low-signal. Take all decisive positions if they fit, then fill with a
    random draw sample."""
    decisive = [it for it in items if it[1] != 0]
    draws = [it for it in items if it[1] == 0]
    rng = random.Random(seed)
    if len(decisive) >= k:
        return rng.sample(decisive, k)
    fill = k - len(decisive)
    chosen_draws = draws if len(draws) <= fill else rng.sample(draws, fill)
    return decisive + chosen_draws


def cross_check(
    all_solved: dict[str, Value],
    max_n: int,
    sample: int = 0,
    seed: int = 0,
    workers: int = 1,
) -> tuple[list[tuple[str, str, str]], int]:
    """Compare TB entries to the oracle. Returns (mismatches, n_checked).

    If `sample` > 0 and the table is larger, cross-check a stratified sample
    (all decisive positions plus a random draw fill) instead of every position
    — the depth-(2N+8) oracle is ~0.8 s/position at N=2, so a full check is
    infeasible. TB values are exact by construction; this only catches bugs.
    `workers` > 1 runs the oracle across processes (pure-CPU, embarrassingly
    parallel)."""
    depth = 2 * max_n + 8
    items = [(fen, wdl, dtm) for fen, (wdl, dtm) in all_solved.items()]
    if sample > 0 and len(items) > sample:
        items = _stratified_sample(items, sample, seed)

    work = [(fen, _tb_black_dtm(fen, wdl, dtm), depth) for fen, wdl, dtm in items]

    mismatches: list[tuple[str, str, str]] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(_check_one, work, chunksize=16):
                if res is not None:
                    mismatches.append(res)
    else:
        for w in work:
            res = _check_one(w)
            if res is not None:
                mismatches.append(res)
    return mismatches, len(work)


# --------------------------------------------------------------------------- #
# Reporting / persistence
# --------------------------------------------------------------------------- #

def _wdl_distribution(table: dict[str, Value]) -> dict[str, int]:
    dist = {"win": 0, "loss": 0, "draw": 0}
    for wdl, _ in table.values():
        if wdl > 0:
            dist["win"] += 1
        elif wdl < 0:
            dist["loss"] += 1
        else:
            dist["draw"] += 1
    return dist


def _persist(tables: dict[int, dict[str, Value]], max_n: int) -> Path:
    TB_DIR.mkdir(parents=True, exist_ok=True)
    merged: dict[str, Value] = {}
    for t in tables.values():
        merged.update(t)
    path = TB_DIR / f"phase1_N{max_n}.pkl"
    with path.open("wb") as f:
        pickle.dump(merged, f)
    return path


def run(max_n: int, sample: int = 0, workers: int = 1) -> dict[str, Value]:
    print(f"=== Solving tablebase: White lone King vs Black total <= {max_n} ===")
    tables = solve_up_to(max_n)
    merged: dict[str, Value] = {}
    for level in range(max_n + 1):
        t = tables[level]
        dist = _wdl_distribution(t)
        print(
            f"level {level}: {len(t):>8} positions  "
            f"win={dist['win']} loss={dist['loss']} draw={dist['draw']}"
        )
        merged.update(t)

    total_dist = _wdl_distribution(merged)
    print(
        f"TOTAL: {len(merged)} positions  "
        f"win={total_dist['win']} loss={total_dist['loss']} draw={total_dist['draw']}"
    )

    path = _persist(tables, max_n)
    print(f"persisted -> {path}")

    scope = (
        f"sampled {sample} (stratified)"
        if sample > 0 and len(merged) > sample
        else "full"
    )
    print(f"cross-checking against endgame_solver oracle ({scope}, workers={workers}) ...")
    mismatches, n_checked = cross_check(merged, max_n, sample=sample, workers=workers)
    if mismatches:
        print(f"cross-check: {n_checked} checked, {len(mismatches)} MISMATCHES")
        for fen, tb, oracle in mismatches:
            print(f"  MISMATCH {fen}")
            print(f"           TB:     {tb}")
            print(f"           oracle: {oracle}")
    else:
        print(f"cross-check: {n_checked} checked, 0 mismatches")
    return merged


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--max-n", type=int, default=None,
                   help="solve only this N (default: run gate N=2 then N=4)")
    p.add_argument("--sample", type=int, default=2000,
                   help="cross-check a stratified sample of this many positions "
                        "(0 = full; full N>=2 is infeasible)")
    p.add_argument("--workers", type=int, default=6,
                   help="parallel oracle workers for the cross-check")
    args = p.parse_args()

    if args.max_n is not None:
        run(args.max_n, sample=args.sample, workers=args.workers)
        return 0

    # Default: gate at N=2, then N=4.
    merged2 = run(2, sample=args.sample, workers=args.workers)
    print()
    print("=== N=4 ===")
    merged4 = run(4, sample=args.sample, workers=args.workers)
    seed = "8/8/8/3kk3/8/8/8/4K3[d5:kk,e5:kk] b - - 0 1"
    key = serialize_fen(parse_fen(seed))
    v = merged4.get(key)
    print(f"curriculum seed {seed}")
    print(f"  canonical key: {key}")
    if v is None:
        print("  -> NOT in N=4 table")
    else:
        wdl, dtm = v
        label = {1: "WIN", 0: "DRAW", -1: "LOSS"}[wdl]
        print(f"  -> TB value (black to move): {label}, dtm={dtm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
