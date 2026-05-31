#!/usr/bin/env python
"""Generate the Phase-1 endgame tablebase onto an external drive.

Climbs material classes 0..max-total (total Black pieces, White = lone King),
solving each into <root>/phase1/N<k>.tb via the parallel fixpoint driver.
Resumable: classes already marked solved in <root>/manifest.json are skipped.

    python scripts/gen_tablebase.py --max-total 4 --workers 6 \
        --root /Volumes/hd0/chessckers_tb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tb import driver  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="/Volumes/hd0/chessckers_tb",
                   help="storage root (must exist / be writable)")
    p.add_argument("--max-total", type=int, required=True,
                   help="highest total Black piece count to solve")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--no-resume", action="store_true",
                   help="re-solve classes even if marked solved")
    args = p.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    driver.generate(root, args.max_total, workers=args.workers,
                    resume=not args.no_resume)
    print(f"done -> {driver.store.manifest_path(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
