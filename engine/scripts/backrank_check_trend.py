"""Is Black learning back-rank checks? Replay PGNs through PyVariant and read the
rank-8 camp counter (FEN '{r8:N}') + check flag each ply.

A White camp turn sets r8>=1 (on White's move). A Black move that leaves White in
check resets r8->0. So on a BLACK ply:
  reset  = r8 was >=1 before, ==0 after  -> Black delivered a camp-denying back-rank check
  br_chk = check flag True AND White king on rank 8 -> any back-rank check
Samples B evenly-spaced buckets across the run to show the trend.
Usage: python cc_checks.py <pgn_dir> [K_per_bucket] [N_buckets]
"""
import sys, glob, re, os
from chessckers_engine.variant_py.client import PyVariantClient

START_FEN = "3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] b - - 0 1"


def r8(fen):
    m = re.search(r'r8:(\d+)', fen)
    return int(m.group(1)) if m else 0


def wk_on_rank8(fen):
    return 'K' in fen.split()[0].split('/')[0]


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


def analyze(path, client):
    line = open(path).read()
    toks = parse_tokens(line)
    if not toks:
        return None
    fen, prev_r8, maxr8 = START_FEN, 0, 0
    r = dict(res=result_token(line), camped=False, resets=0, br_chk=0, maxr8=0)
    for i, tok in enumerate(toks, 1):
        try:
            gs = client.make_move(fen, tok)
        except Exception:
            return None
        fen = gs["fen"]
        cur = r8(fen)
        maxr8 = max(maxr8, cur)
        if cur >= 1:
            r["camped"] = True
        if i % 2 == 1:  # Black ply
            if prev_r8 >= 1 and cur == 0:
                r["resets"] += 1
            if gs.get("check") and wk_on_rank8(fen):
                r["br_chk"] += 1
        prev_r8 = cur
        if gs.get("status"):
            break
    r["maxr8"] = maxr8
    return r


def pct(n, d):
    return f"{100*n/d:4.1f}%" if d else " n/a "


def summarize(label, rows):
    rows = [r for r in rows if r]
    n = len(rows)
    if not n:
        print(f"{label}: 0 games")
        return
    w = sum(r["res"] == "1-0" for r in rows)
    b = sum(r["res"] == "0-1" for r in rows)
    dr = sum(r["res"] == "1/2-1/2" for r in rows)
    camped = [r for r in rows if r["camped"]]
    c = len(camped)
    reset_ge1 = sum(r["resets"] >= 1 for r in camped)
    mean_resets = sum(r["resets"] for r in camped) / c if c else 0
    anychk = sum(r["br_chk"] >= 1 for r in rows)
    print(f"{label}: n={n} | W {pct(w,n)} B {pct(b,n)} d {pct(dr,n)} "
          f"| White camps {pct(c,n)} | of camps, Black checks-to-reset >=1: {pct(reset_ge1,c)} "
          f"(mean {mean_resets:.2f}) | any back-rank check {pct(anychk,n)}")


def main():
    d = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    files = glob.glob(os.path.join(d, "*.pgn"))
    num = lambda q: int(re.search(r'(\d+)\.pgn$', q).group(1)) if re.search(r'(\d+)\.pgn$', q) else -1
    files = sorted(files, key=num)
    N = len(files)
    client = PyVariantClient()
    print(f"{N} pgns (#{num(files[0])}..#{num(files[-1])}); {B} buckets x {K} games\n")
    for k in range(B):
        center = int((k + 0.5) / B * N)
        lo = max(0, center - K // 2)
        chunk = files[lo:lo + K]
        label = f"#{num(chunk[0]):>6}-{num(chunk[-1]):>6}"
        summarize(label, [analyze(f, client) for f in chunk])


if __name__ == "__main__":
    main()
