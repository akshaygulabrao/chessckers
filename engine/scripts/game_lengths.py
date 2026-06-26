#!/usr/bin/env python3
"""game_lengths — average chessckers game length over training.

Runs ON the box (via `cc lengths`). Groups PGN files into windows and plots the
average game length per window so you can see the learning curve:

  • Early training: length increases — Black learns to avoid blunders, survives longer.
  • Mid/late training: length decreases sharply — Black learns to deliver mate
    (capture White's king) instead of just surviving.

  cc lengths                  # 50-game windows, last 80 windows
  cc lengths --window 100     # 100-game windows
  cc lengths --all            # all games from the start
  cc lengths --loop 30        # refresh every 30s (watch live)

Output columns:
  block     first game in window
  n         games in window
  avg_ply   average full-move count per game
  p50       median
  trend     arrow: → (flat), ↗ (rising), ↘ (falling)
"""

import argparse
import re
import time
from pathlib import Path

BARS = "▁▂▃▄▅▆▇█"
ARROWS = {">0": "↗", "<0": "↘", "=0": "→"}

_RESULT_RE = re.compile(r"\b(1-0|0-1|1/2-1/2)\b")
_COMMENT_RE = re.compile(r"\{[^}]*\}")  # {1} or {OL: 0}


def _parse_pgn_ply(line: str) -> int:
    """Count move tokens in one PGN line. Each token that is not a comment or
    result is one half-move (ply)."""
    line = _COMMENT_RE.sub("", line)
    # Remove the result token if present
    line = _RESULT_RE.sub("", line)
    return len(line.split())


def _parse_pgns_dir(pgn_dir: Path) -> list[tuple[int, int]]:
    """Read all .pgn files sorted by game number, return [(game#, ply), ...]."""
    files = sorted(pgn_dir.glob("*.pgn"), key=lambda p: int(p.stem))
    data: list[tuple[int, int]] = []
    for f in files:
        try:
            text = f.read_text().strip()
        except OSError:
            continue
        if not text:
            continue
        ply = _parse_pgn_ply(text)
        data.append((int(f.stem), ply))
    return data


def _percentile(sorted_vals: list[int], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return float(sorted_vals[idx])


def _spark(vals: list[float], lo: float | None = None, hi: float | None = None) -> str:
    if not vals:
        return "(no data)"
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    rng = (hi - lo) or 1.0
    return "".join(BARS[min(7, max(0, int((v - lo) / rng * 7.999)))] for v in vals)


def _window_data(data: list[tuple[int, int]], window: int) -> list[dict]:
    """Group (game#, ply) pairs into windows of `window` games."""
    if not data:
        return []
    windows: list[dict] = []
    lo = data[0][0]
    hi = data[-1][0]
    for start in range(lo, hi + 1, window):
        block: list[int] = [p for g, p in data if start <= g < start + window]
        if not block:
            continue
        avg = sum(block) / len(block)
        block.sort()
        windows.append(
            {
                "block": start,
                "n": len(block),
                "avg": avg,
                "p50": _percentile(block, 0.50),
            }
        )
    return windows


def _trend_arrow(avgs: list[float], idx: int) -> str:
    """Compare this window's avg to the previous; → if first window."""
    if idx == 0:
        return "·"
    diff = avgs[idx] - avgs[idx - 1]
    key = f"{diff:+.1f}"
    sign_key = f"{'>0' if diff > 0.5 else '<0' if diff < -0.5 else '=0'}"
    return ARROWS.get(sign_key, "→")


def main():
    parser = argparse.ArgumentParser(description="Average game length over training")
    parser.add_argument("--root", default="/workspace/chessckers/lczero-server")
    parser.add_argument("--run", default=None)
    parser.add_argument("--window", type=int, default=50, help="games per window")
    parser.add_argument(
        "--tail", type=int, default=80, help="last N windows to show (0=all)"
    )
    parser.add_argument(
        "--all", action="store_true", help="show all windows from the start"
    )
    parser.add_argument("--loop", type=float, default=0.0, help="refresh every N s")
    args = parser.parse_args()

    # Resolve run dir — latest if not specified
    if args.run:
        pgn_dir = Path(args.root) / "pgns" / args.run
    else:
        candidates = sorted(
            [d for d in (Path(args.root) / "pgns").iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print("game_lengths: no pgns/ run dirs under", Path(args.root) / "pgns")
            return 1
        pgn_dir = candidates[0]

    tail = 0 if args.all else args.tail

    while True:
        data = _parse_pgns_dir(pgn_dir)
        windows = _window_data(data, args.window)

        if not windows:
            print("(no PGN files yet)")
        else:
            # Collect averages for sparklines and trend arrows
            avgs = [w["avg"] for w in windows]
            lo = min(avgs)
            hi = max(avgs)

            print(
                f"=== game length over {len(windows)} windows"
                f" ({len(data)} games, window={args.window}) "
                f"{time.strftime('%H:%M:%S')} ==="
            )
            print(
                f"  min={lo:.0f}  max={hi:.0f}  "
                f"latest avg={avgs[-1]:.0f}  latest p50={windows[-1]['p50']:.0f}"
            )
            print(f"  sparkline:  {_spark(avgs)}")
            print(f"  {'block':>6}  {'n':>5}  {'avg':>6}  {'p50':>5}  {'trend':>5}")
            print(f"  {'─' * 6}  {'─' * 5}  {'─' * 6}  {'─' * 5}  {'─' * 5}")

            show = windows[-tail:] if tail else windows
            for i, w in enumerate(show):
                real_idx = len(windows) - len(show) + i
                arrow = _trend_arrow(avgs, real_idx)
                print(
                    f"  {w['block']:>6}  {w['n']:>5}  {w['avg']:>6.1f}  "
                    f"{w['p50']:>5.0f}  {arrow:>5}"
                )

            # Highlight inflection point — where the trend switches from rising to falling
            rising = [
                (i, avgs[i]) for i in range(1, len(avgs)) if avgs[i] > avgs[i - 1] + 0.5
            ]
            falling = [
                (i, avgs[i]) for i in range(1, len(avgs)) if avgs[i] < avgs[i - 1] - 0.5
            ]
            last_rising = rising[-1][0] if rising else 0
            first_falling = falling[0][0] if falling else len(avgs)
            if first_falling > last_rising and first_falling < len(avgs):
                print(
                    f"\n  Peak avg {max(avgs):.0f} plies → "
                    f"then falling trend: Black may be learning to mate"
                )

        if not args.loop:
            return 0
        time.sleep(args.loop)
        print("\033[2J\033[H", end="")


if __name__ == "__main__":
    raise SystemExit(main())
