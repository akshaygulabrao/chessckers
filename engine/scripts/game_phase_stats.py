"""Replay early/mid/late selfplay PGNs through PyVariant and characterize them.

Black MATERIAL = sum of stack heights in the FEN overlay (start = 4: d8:kk + e8:kk).
Material only changes on captures/suicides (deploys/stacking just rearrange it), so:
  material drop after a BLACK move  = Black self-destruct (a ram/suicide)
  material drop after a WHITE move  = White captured Black material
Usage: python cc_analyze.py <pgn_dir> [K_per_bucket]
"""
import sys, glob, re, os
from chessckers_engine.variant_py.client import PyVariantClient

START_FEN = "3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] b - - 0 1"
SIMPLE = re.compile(r'^([a-h][1-8])([a-h][1-8])')
CAD = re.compile(r'^c\d+:([a-h][1-8])~.*->([a-h][1-8])')


def black_material(fen):
    bf = fen.split()[0]
    if '[' not in bf:
        return 0
    overlay = bf[bf.index('[') + 1: bf.rindex(']')]
    return sum(len(e.split(':', 1)[1]) for e in overlay.split(',') if ':' in e)


def black_piles(fen):  # occupied Black squares (towers/piles), for "fully eliminated"
    return sum(1 for ch in fen.split()[0].split('[')[0] if ch in "pnbrqk")


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


def result_token(line):
    for t in line.split():
        if t in ("1-0", "0-1", "1/2-1/2"):
            return t
    return None


def from_to(tok):
    m = SIMPLE.match(tok) or CAD.match(tok)
    return (m.group(1), m.group(2)) if m else (None, None)


def analyze(path, client):
    line = open(path).read()
    toks = parse_tokens(line)
    if not toks:
        return None
    fen = START_FEN
    prev = black_material(fen)
    r = dict(plies=0, res=result_token(line), status="", winner="", mat_end=prev,
             suicide=0, whitecap=0, suicide_r8=0, charge_r8=0, ever_check=False)
    for i, tok in enumerate(toks, 1):
        try:
            gs = client.make_move(fen, tok)
        except Exception:
            r["bad"] = tok
            return r
        fen = gs["fen"]
        r["plies"] = i
        cur = black_material(fen)
        black_moved = (i % 2 == 1)  # Black is to move at the start, so odd plies = Black
        if cur < prev:
            d = prev - cur
            if black_moved:
                r["suicide"] += d
                fr, to = from_to(tok)
                if to and to[1] == '8':
                    r["suicide_r8"] += 1
                    if fr and fr[1] == '8':
                        r["charge_r8"] += 1
            else:
                r["whitecap"] += d
        if black_moved and gs.get("check"):
            r["ever_check"] = True
        prev = cur
        if gs.get("status"):
            r["status"], r["winner"] = gs["status"], gs.get("winner", "")
            break
    r["mat_end"] = prev
    r["piles_end"] = black_piles(fen)
    return r


def pct(n, d):
    return f"{100*n/d:4.1f}%" if d else "  n/a"


def summarize(label, rows):
    rows = [r for r in rows if r]
    bad = [r for r in rows if r.get("bad")]
    rows = [r for r in rows if not r.get("bad")]
    n = len(rows)
    if not n:
        print(f"\n## {label}: 0 usable games ({len(bad)} unparseable)")
        return
    plies = sorted(r["plies"] for r in rows)
    p = lambda i: plies[min(int(i * (len(plies) - 1)), len(plies) - 1)]
    w = sum(r["res"] == "1-0" for r in rows)
    b = sum(r["res"] == "0-1" for r in rows)
    dr = sum(r["res"] == "1/2-1/2" for r in rows)
    wwins = [r for r in rows if r["res"] == "1-0"]
    elim = sum(r.get("piles_end", 1) == 0 for r in wwins)
    mat_mean = sum(r["mat_end"] for r in rows) / n
    keep_all = sum(r["mat_end"] == 4 for r in rows)
    g_suicide = sum(r["suicide"] > 0 for r in rows)
    g_whitecap = sum(r["whitecap"] > 0 for r in rows)
    g_check = sum(r["ever_check"] for r in rows)
    g_sr8 = sum(r["suicide_r8"] > 0 for r in rows)
    g_cr8 = sum(r["charge_r8"] > 0 for r in rows)
    print(f"\n## {label}: {n} games  ({len(bad)} unparseable)")
    print(f"  result          W {pct(w,n)}  B {pct(b,n)}  draw {pct(dr,n)}")
    print(f"  game length     p10={p(.1)}  p50={p(.5)}  p90={p(.9)}  p99={p(.99)}  max={plies[-1]}")
    print(f"  White wins by   rank-8 camp {len(wwins)-elim}/{len(wwins)}   vs eliminating all Black {elim}/{len(wwins)}")
    print(f"  Black ever delivers a check     {pct(g_check,n)}  ({g_check}/{n})")
    print(f"  Black material left (of 4)      mean {mat_mean:.2f}   keeps all 4 in {pct(keep_all,n)}")
    print(f"  Black throws away material (suicide/ram)  {pct(g_suicide,n)} of games   [{sum(r['suicide'] for r in rows)} pieces]")
    print(f"  Black material taken by White             {pct(g_whitecap,n)} of games   [{sum(r['whitecap'] for r in rows)} pieces]")
    print(f"  Black self-kills onto rank 8    {pct(g_sr8,n)} of games   (orthogonal charge along rank 8: {pct(g_cr8,n)})")


def main():
    d = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    files = glob.glob(os.path.join(d, "*.pgn"))
    num = lambda q: int(re.search(r'(\d+)\.pgn$', q).group(1)) if re.search(r'(\d+)\.pgn$', q) else -1
    files = sorted(files, key=num)
    N = len(files)
    print(f"{N} pgns total (#{num(files[0])}..#{num(files[-1])}); sampling {K}/bucket")
    buckets = {
        f"EARLY (#{num(files[0])}..)": files[:K],
        f"MID (~#{num(files[N//2])})": files[N // 2 - K // 2: N // 2 + K // 2],
        f"LATE (..#{num(files[-1])})": files[-K:],
    }
    client = PyVariantClient()
    for label, flist in buckets.items():
        summarize(label, [analyze(f, client) for f in flist])


if __name__ == "__main__":
    main()
