#!/usr/bin/env python3
"""Generate a PyVariant oracle corpus for the C++ rules-parity test.

PyVariant is the Chessckers rules authority; the akshay-chessckers-0 fork's C++
port (gen_legal_native / detect_status / apply) must match it exactly. This
plays deterministic random games and records, for every position reached:

    {"fen": ..., "legal": [uci, ...sorted], "status": ..., "winner": ...}

The C++ parity_test.cc loads the JSONL and asserts its own move-gen + status
agree. Regenerate after any rule change:

    .venv/bin/python scripts/gen_parity_corpus.py \
        ../akshay-chessckers-0/src/chessckers/corpus/parity_corpus.jsonl

Determinism: seeded RNG + fixed traversal, and (per the env note) Chessckers
move-gen is deterministic, so the corpus is reproducible.
"""
from __future__ import annotations

import json
import random
import sys

from chessckers_engine.variant_py import PyVariantClient

N_GAMES = 400
MAX_PLIES = 80
MAX_POSITIONS = 8000  # cap so the committed corpus stays a fixed, reproducible size
SEED = 0

# Known terminal positions (mirrors the C++ CcTerminal cases) so every terminal
# type is covered even if random games rarely reach them.
SEED_FENS = [
    "8/8/8/8/7K/2P5/1P6/p7[a1:s] b - - 0 1",        # Black self-stalemate -> White
    "7K/5k2/8/8/8/8/8/8[f7:kkk] w - - 0 1",          # White stalemate -> draw
    "7K/5kk1/8/8/8/8/8/8[f7:kk,g7:kk] w - - 0 1",    # White mated -> Black
    "8/8/8/8/7K/8/8/8 b - - 0 1",                     # Black eliminated -> White
    "8/8/8/8/8/8/8/p7[a1:s] w - - 0 1",              # White king captured -> Black
    "3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] b - - 0 1",   # version_5 start
]


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "parity_corpus.jsonl"
    client = PyVariantClient()
    rng = random.Random(SEED)
    seen: set[str] = set()
    records: list[dict] = []

    def record(d: dict) -> None:
        fen = d["fen"]
        if fen in seen:
            return
        seen.add(fen)
        records.append({
            "fen": fen,
            "legal": sorted(m["uci"] for m in d["legalMoves"]),
            "status": d["status"] or "",
            "winner": d["winner"] or "",
        })

    for fen in SEED_FENS:
        record(client.new_game(fen))

    for _ in range(N_GAMES):
        if len(records) >= MAX_POSITIONS:
            break
        d = client.new_game()
        for _ in range(MAX_PLIES):
            record(d)
            if d["status"] or not d["legalMoves"] or len(records) >= MAX_POSITIONS:
                break
            uci = rng.choice(d["legalMoves"])["uci"]
            d = client.make_move(d["fen"], uci)

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")

    n_terminal = sum(1 for r in records if r["status"])
    print(f"wrote {len(records)} unique positions ({n_terminal} terminal) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
