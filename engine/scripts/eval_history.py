"""Undertrained vs. stuck: replay the WHOLE checkpoint history on a frozen probe
suite and reconstruct the strength-vs-time curve from data already on disk.

For each iter-async-*.pt (+ the live weights.pt), score the mean POLICY mass and
MCTS-visit mass the net puts on the correct (camp-denying check) move set, plus
argmax hit-rate. Because the suite + settings are frozen (temp 0, noise off, fixed
seed), the only variable is the weights:

  * mass still RISING at the latest checkpoints  -> still learning => UNDERTRAINED
  * mass PLATEAUED many checkpoints ago          -> converged    => STUCK

Run --sims 2 for the (near-free) raw-policy curve over every checkpoint; raise
--sims (+ --stride to subsample) for the search-amplified curve. Policy mass is
the load-bearing signal: a deep, high-branching forced win is only reachable via
a learned policy, so policy concentration -- not search depth -- is what moves.

Usage: python eval_history.py <ckpt_dir> <suite.jsonl> [--sims 2] [--stride 1]
                              [--device cuda] [--out hist.jsonl]
"""
import argparse, glob, json, os, re, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import watch_game  # noqa: E402  (checkpoint resolver + START_FEN)


def mass_on(root, ucis):
    prior = sum(c.prior for u, c in root.children.items() if u in ucis)
    tv = sum(c.visits for c in root.children.values()) or 1
    vis = sum(c.visits for u, c in root.children.items() if u in ucis) / tv
    return prior, vis


def pol_pick(root):
    return max(root.children.values(), key=lambda c: c.prior).move_to_here["uci"]


def start_value(model, device):
    """Best-effort: net WDL on the start FEN (Black to move). None if unavailable."""
    try:
        import torch
        from chessckers_engine.encoding import encoders_for
        enc_pos, _, _ = encoders_for(getattr(model, "VERSION", "v1"))
        pos = enc_pos(watch_game.DEFAULT_START_FEN).unsqueeze(0).to(device)
        with torch.no_grad():
            wdl = torch.softmax(model.value(pos).reshape(-1)[:3], dim=-1)
        return [round(float(x), 3) for x in wdl]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt_dir")
    ap.add_argument("suite")
    ap.add_argument("--sims", type=int, default=2)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import torch
    from chessckers_engine.checkpoints import load_scorer
    from chessckers_engine.mcts_puct import run_mcts
    from chessckers_engine.variant_py.client import PyVariantClient

    suite = [json.loads(l) for l in open(a.suite)]
    for e in suite:
        e["best"] = set(e["best"])
    idx = lambda p: int(re.search(r'(\d+)\.pt$', p).group(1)) if re.search(r'(\d+)\.pt$', p) else -1
    cks = sorted(glob.glob(os.path.join(a.ckpt_dir, "iter-async-*.pt")), key=idx)[::a.stride]
    live = os.path.join(a.ckpt_dir, "weights.pt")
    if os.path.exists(live):
        cks.append(live)
    client = PyVariantClient()
    out = open(a.out, "w") if a.out else None
    print(f"{len(cks)} checkpoints x {len(suite)} probes @ sims={a.sims}, device={a.device}\n")
    print(f"{'checkpoint':>22} {'polmass':>8} {'vismass':>8} {'pol_acc':>8} {'mcts_acc':>8} {'start WDL(B)':>16}")
    for c in cks:
        try:
            model = load_scorer(c).to(a.device).eval()
        except Exception as ex:
            print(f"{os.path.basename(c):>22}  skip: {ex}")
            continue
        pm = vm = pa = ma = 0.0
        for e in suite:
            st = client.new_game(e["fen"])
            r = run_mcts(st, client, model, n_sims=max(2, a.sims), c_puct=a.c_puct, dirichlet_alpha=None)
            p, v = mass_on(r.root, e["best"])
            pm += p; vm += v
            pa += pol_pick(r.root) in e["best"]
            ma += r.chosen["uci"] in e["best"]
        n = len(suite) or 1
        sv = start_value(model, a.device)
        row = dict(ckpt=os.path.basename(c), idx=idx(c), sims=a.sims, n=len(suite),
                   polmass=round(pm / n, 4), vismass=round(vm / n, 4),
                   pol_acc=round(pa / n, 4), mcts_acc=round(ma / n, 4), start_wdl=sv)
        print(f"{row['ckpt']:>22} {row['polmass']:>8.3f} {row['vismass']:>8.3f} "
              f"{row['pol_acc']:>8.3f} {row['mcts_acc']:>8.3f} {str(sv):>16}")
        if out:
            out.write(json.dumps(row) + "\n"); out.flush()
        del model
        if a.device == "cuda":
            torch.cuda.empty_cache()
    if out:
        out.close()


if __name__ == "__main__":
    main()
