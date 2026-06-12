#!/usr/bin/env python
"""Fixed eval bench: score a net on a suite of Black-winning positions with a
known best move. For each position it reports (a) the raw network policy pick
(no search) and (b) the MCTS pick at --sims, plus how much policy/visit mass
sits on the correct move — so you can watch the net learn (mass shifting onto
the right move) even before the argmax flips.

  cd engine
  .venv/bin/python scripts/eval_bench.py                       # latest net, 200 sims
  .venv/bin/python scripts/eval_bench.py --weights weights/run/weights.pt --sims 400
  .venv/bin/python scripts/eval_bench.py --device cpu --sims 1  # raw policy only

Add positions to POSITIONS below. `best` is the set of acceptable UCIs (use the
EXACT uci PyVariant emits, incl. any [n]/cadence annotation — `d3e2` mates but
`d3e2[1]` only stalemates, so the annotation matters). Ground-truth a new entry
with: client.new_game(fen) then make_move to confirm it's mate/variantEnd/black.
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # so we can reuse watch_game's checkpoint resolver
import watch_game  # noqa: E402

# ---------------------------------------------------------------------------
# The suite. Each entry: name, FEN (Black to move), best = acceptable UCI set.
# ---------------------------------------------------------------------------
POSITIONS: list[dict] = [
    {
        "name": "mate-in-1: d3 tower captures to e2",
        "fen": "8/8/8/8/8/3kk3/8/4K3[d3:kk,e3:kk] b - - 0 1",
        "best": ["d3e2"],  # NOT d3e2[1] (deploy-only -> stalemate)
    },
]


def _policy_pick(root) -> tuple[str, float]:
    """Most-likely move under the raw network policy (max prior), and the prior
    mass that pick carries. Independent of search/visits."""
    best = max(root.children.values(), key=lambda c: c.prior)
    return best.move_to_here["uci"], best.prior


def _mass_on(root, ucis: set[str]) -> tuple[float, float]:
    """(policy prior mass, visit share) sitting on the acceptable-move set."""
    prior = sum(c.prior for u, c in root.children.items() if u in ucis)
    total_v = sum(c.visits for c in root.children.values()) or 1
    visit = sum(c.visits for u, c in root.children.items() if u in ucis) / total_v
    return prior, visit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="", help="checkpoint .pt (default: latest published net)")
    ap.add_argument("--latest", action="store_true", help="prefer the fleet's live weights.pt")
    ap.add_argument("--sims", type=int, default=200, help="MCTS sims/position (default 200; 1 = raw policy)")
    ap.add_argument("--device", default="cpu", help="cpu|mps|cuda")
    ap.add_argument("--c-puct", type=float, default=1.5)
    args = ap.parse_args()

    from chessckers_engine.checkpoints import load_scorer
    from chessckers_engine.mcts_puct import run_mcts
    from chessckers_engine.variant_py.client import PyVariantClient

    cands = watch_game._resolve_weights(args.weights, latest=args.latest)
    model = None
    for cand in cands:
        try:
            model = load_scorer(cand).to(args.device).eval()
            print(f"net: {cand}   device={args.device}   sims={args.sims}\n")
            break
        except Exception as e:  # noqa: BLE001
            print(f"  (skip {cand}: {e})")
    if model is None:
        raise SystemExit("could not load any candidate checkpoint")

    client = PyVariantClient()
    pol_ok = mcts_ok = 0
    header = f"{'position':<40} {'pol pick':<12} {'mcts pick':<12} {'pol%':>5} {'vis%':>5}"
    print(header)
    print("-" * len(header))
    for pos in POSITIONS:
        best = set(pos["best"])
        state = client.new_game(pos["fen"])
        # Noise OFF (alpha=None) -> deterministic eval.
        result = run_mcts(state, client, model, n_sims=max(2, args.sims),
                          c_puct=args.c_puct, dirichlet_alpha=None)
        pol_pick, _ = _policy_pick(result.root)
        mcts_pick = result.chosen["uci"]
        prior_mass, visit_mass = _mass_on(result.root, best)
        p_ok = pol_pick in best
        m_ok = mcts_pick in best
        pol_ok += p_ok
        mcts_ok += m_ok
        print(f"{pos['name']:<40} "
              f"{(pol_pick + (' OK' if p_ok else '')):<12} "
              f"{(mcts_pick + (' OK' if m_ok else '')):<12} "
              f"{100*prior_mass:>4.0f}% {100*visit_mass:>4.0f}%")

    n = len(POSITIONS)
    print("-" * len(header))
    print(f"policy argmax: {pol_ok}/{n}   mcts argmax: {mcts_ok}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
