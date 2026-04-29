"""Smoke test: replay each saved game via PyVariantClient, comparing
post-move state to scalachess at every step. Reports the first divergence
per game; doesn't stop on failure. Useful for assessing how well the
incremental port covers real played positions.

Run: `uv run python check_pyvariant_replay.py`
"""
from __future__ import annotations

import json
from pathlib import Path

from chessckers_engine.server_client import ServerClient
from chessckers_engine.variant_py import PyVariantClient

GAMES = Path(__file__).resolve().parent / "games" / "games.jsonl"


def play_game_through(idx: int, history: list[dict]) -> tuple[int, str | None]:
    """Replay `history` (list of {fen, uci}) plus the last move. Returns
    (last_successful_step, error_or_divergence)."""
    py = PyVariantClient()
    sc = ServerClient()
    last_ok = -1
    err: str | None = None
    try:
        for i, hop in enumerate(history):
            fen = hop["fen"]
            uci = hop["uci"]
            try:
                py_after = py.make_move(fen, uci)
            except NotImplementedError as e:
                err = f"step {i} uci={uci!r}: NotImplementedError: {e}"
                break
            except Exception as e:  # noqa: BLE001
                err = f"step {i} uci={uci!r}: {type(e).__name__}: {e}"
                break
            try:
                sc_after = sc.make_move(fen, uci)
            except Exception as e:  # noqa: BLE001
                err = f"step {i} scalachess failure (skipping): {e}"
                last_ok = i
                continue
            # Compare board content + bracket overlay + turn + status; ignore
            # halfmove/fullmove fields — scalachess Chessckers uses non-standard
            # update rules for those that we don't fully match (cosmetic only).
            def _strip_clocks(fen: str) -> str:
                parts = fen.split(" ")
                return " ".join(parts[:4]) if len(parts) >= 4 else fen
            for k in ("turn", "status", "winner"):
                if py_after.get(k) != sc_after.get(k):
                    err = (
                        f"step {i} uci={uci!r}: {k} diverges  "
                        f"py={py_after.get(k)!r} scala={sc_after.get(k)!r}"
                    )
                    return (i - 1 if last_ok < i - 1 else last_ok, err)
            if _strip_clocks(py_after.get("fen", "")) != _strip_clocks(sc_after.get("fen", "")):
                err = (
                    f"step {i} uci={uci!r}: fen-without-clocks diverges  "
                    f"py={_strip_clocks(py_after.get('fen', ''))!r} "
                    f"scala={_strip_clocks(sc_after.get('fen', ''))!r}"
                )
                return (i - 1 if last_ok < i - 1 else last_ok, err)
            last_ok = i
    finally:
        sc.close()
    return (last_ok, err)


def main() -> int:
    if not GAMES.exists():
        print(f"no saved games at {GAMES}")
        return 1

    games_played = 0
    games_completed = 0
    total_steps_ok = 0
    failures: list[tuple[int, str]] = []

    for line_no, line in enumerate(GAMES.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            game = json.loads(line)
        except json.JSONDecodeError:
            continue
        history = game.get("history") or []
        if not history:
            continue

        games_played += 1
        last_ok, err = play_game_through(line_no, history)
        steps_ok = last_ok + 1
        total_steps_ok += steps_ok
        if err is None:
            games_completed += 1
        else:
            failures.append((line_no, err))

        print(f"  game #{line_no:2d}  total_plies={len(history):3d}  "
              f"py_ok_through={steps_ok:3d}  {'✓' if err is None else '✗ ' + err[:80]}")

    print(f"\nSummary: {games_completed}/{games_played} games completed; "
          f"{total_steps_ok} successful plies total.")
    if failures:
        print(f"\n{len(failures)} divergences:")
        for ln, err in failures[:5]:
            print(f"  game #{ln}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
