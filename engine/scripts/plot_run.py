#!/usr/bin/env python3
"""plot_run — terminal sparklines of a run's curves (no matplotlib needed).

Runs ON the box (via `cc plot`). Shows the two things you actually watch:
  * Black-win % over game-blocks  (the convergence curve)
  * policy_top1_agree & value_sign_agree over training steps (from train_metrics.jsonl)
Also plots run_metrics.csv if present (the run_doctor --csv sampler output).

  cc plot                  # default curves
  cc plot --block 2000     # game-block size for the win-rate curve
"""
import argparse, json, os
import run_doctor as rd

BARS = "▁▂▃▄▅▆▇█"


def spark(vals, lo=None, hi=None):
    vals = [float(v) for v in vals if v is not None and v != ""]
    if not vals:
        return "(no data)"
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    rng = (hi - lo) or 1.0
    return "".join(BARS[min(7, max(0, int((v - lo) / rng * 7.999)))] for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=rd.SERVER)
    ap.add_argument("--run", default=None)
    ap.add_argument("--block", type=int, default=2000)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    run = a.run or rd.latest_run(a.root)

    rows = rd.trend(f"{a.root}/pgns/{run}", a.block)
    if rows:
        bwin = [r[3] for r in rows]
        print(f"Black-win % over {len(rows)} blocks of {a.block} games  (0–100%):")
        print(f"  {spark(bwin, 0, 100)}")
        print(f"  first #{int(rows[0][0])}: B{rows[0][3]:.0f}%   last #{int(rows[-1][0])}: B{rows[-1][3]:.0f}%")

    mf = f"{a.root}/trainer/{run}/train_metrics.jsonl"
    if os.path.exists(mf):
        m = [json.loads(l) for l in open(mf)]
        m = m[:: max(1, len(m) // 80)]
        print(f"\ntraining signals over {len(m)} samples (steps {int(m[0]['steps'])}→{int(m[-1]['steps'])}):")
        print(f"  policy_top1_agree  {spark([r['policy_top1_agree'] for r in m], 0, 1)}  "
              f"{m[0]['policy_top1_agree']:.2f}→{m[-1]['policy_top1_agree']:.2f}")
        print(f"  value_sign_agree   {spark([r['value_sign_agree'] for r in m], 0, 1)}  "
              f"{m[0]['value_sign_agree']:.2f}→{m[-1]['value_sign_agree']:.2f}")

    csv = a.csv or f"{a.root}/trainer/{run}/run_metrics.csv"
    if os.path.exists(csv):
        import csv as _csv
        rdr = list(_csv.DictReader(open(csv)))
        if rdr:
            print(f"\nrun_metrics.csv ({len(rdr)} samples):")
            for col in ("B", "ptop1", "games_per_s"):
                if col in rdr[0]:
                    print(f"  {col:<12} {spark([r[col] for r in rdr])}")


if __name__ == "__main__":
    main()
