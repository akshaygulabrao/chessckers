"""Mine a frozen 'camp-defense' probe suite for eval_history.py.

Each probe = a position where White is camping (r8>=1), Black to move, with at
least one CHECKING move available. A Black check resets White's rank-8 counter,
so it is the defensive necessity at that position; the 'best' set is exactly
those checking moves -- PyVariant-derived ground truth, no solver needed.

Freeze ONCE, then eval_history.py scores every checkpoint on this fixed yardstick
to reconstruct the true strength-vs-time curve (rising at the end => undertrained;
long plateau => stuck).

Usage: python gen_probe_suite.py <pgn_dir> <out.jsonl> [N=40] [game_stride=137]
"""
import sys, glob, re, os, json
from collections import Counter
from chessckers_engine.variant_py.client import PyVariantClient

START_FEN = "3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] b - - 0 1"


def parse_tokens(line):
    line = line.strip()
    if line.startswith("PGN:"):
        line = line[4:]
    out = []
    for t in line.split():
        if t in ("1-0", "0-1", "1/2-1/2", "*") or t.startswith("{"):
            break
        out.append(t)
    return out


def r8(fen):
    m = re.search(r'r8:(\d+)', fen)
    return int(m.group(1)) if m else 0


def checking_moves(client, fen, state):
    """UCIs whose application leaves White in check (i.e. resets r8)."""
    best = []
    for m in state.get("legalMoves") or []:
        try:
            res = client.make_move(fen, m["uci"])
        except Exception:
            continue
        if res.get("check"):
            best.append(m["uci"])
    return best


def main():
    d, out = sys.argv[1], sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    stride = int(sys.argv[4]) if len(sys.argv) > 4 else 137
    files = glob.glob(os.path.join(d, "*.pgn"))
    num = lambda q: int(re.search(r'(\d+)\.pgn$', q).group(1)) if re.search(r'(\d+)\.pgn$', q) else -1
    files = sorted(files, key=num)[::stride]  # spread across the run for diversity
    client = PyVariantClient()
    seen, suite = set(), []
    for f in files:
        if len(suite) >= N:
            break
        fen = START_FEN
        for i, tok in enumerate(parse_tokens(open(f).read()), 1):
            try:
                gs = client.make_move(fen, tok)
            except Exception:
                break
            fen = gs["fen"]
            if gs.get("status"):
                break
            # after WHITE's move (even ply): camping + Black to move?
            if i % 2 == 0 and r8(fen) >= 1 and gs.get("turn") == "black" and fen not in seen:
                best = checking_moves(client, fen, gs)
                if best:
                    seen.add(fen)
                    suite.append({"name": f"camp r8={r8(fen)} #{num(f)}@{i}",
                                  "fen": fen, "best": sorted(set(best)), "r8": r8(fen)})
                    if len(suite) >= N:
                        break
    with open(out, "w") as fo:
        for e in suite:
            fo.write(json.dumps(e) + "\n")
    print(f"wrote {len(suite)} probes -> {out}")
    print("  r8 dist:", dict(sorted(Counter(e["r8"] for e in suite).items())))
    if suite:
        nb = sorted(len(e["best"]) for e in suite)
        print(f"  #checking-moves: min={nb[0]} median={nb[len(nb)//2]} max={nb[-1]}")


if __name__ == "__main__":
    main()
