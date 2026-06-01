#!/usr/bin/env python
"""Single-thread VECTORIZED PUCT MCTS: K games in lockstep, one batched NN eval
per macro-step.

Motivation: the per-game `run_mcts` evaluates one leaf per simulation (batch=1).
On MPS a batch-1 forward is ~3.5x slower than on CPU, so the production fleet
runs N CPU processes (batch-1 each). This prototype instead interleaves K
independent games in ONE thread: each macro-step selects one leaf per game and
evaluates all K leaves in a SINGLE `model.batch_eval` call (guaranteed batch=K,
no threads/GIL/IPC/futures/timeout — unlike InferenceServer). That fills the
MPS batch cheaply, exploiting its high large-batch throughput.

Built directly on the production tree primitives (`_select_to_leaf`,
`_expand_with_priors`, `_backup`, `_terminal_value`) so selection/expansion/
backup are IDENTICAL to `run_mcts`; only the eval is hoisted out and batched.
At K=1 with no Dirichlet noise this is bit-for-bit equivalent to `run_mcts`
(see verify_equiv below) — `batch_eval` is documented equal to per-position eval.
"""
from __future__ import annotations

import argparse
import random
import time
from typing import Any

import torch

from chessckers_engine.encoding import encode_move, encode_position_state
from chessckers_engine.mcts_puct import (
    MctsResult,
    PuctNode,
    _apply_dirichlet_noise,
    _backup,
    _expand_with_priors,
    _select_to_leaf,
    _terminal_value,
)
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.variant_py import PyVariantClient

GameState = dict[str, Any]


def _eval_expand_backup(eval_items, client, model, device) -> None:
    """Batched eval of K leaves, then per-leaf expand (with the batch's priors)
    + backup. eval_items: list of (path, leaf, value_only)."""
    if not eval_items:
        return
    positions = torch.stack(
        [encode_position_state(leaf.state) for _, leaf, _ in eval_items]
    ).to(device)
    moves_list: list[torch.Tensor | None] = []
    for _, leaf, value_only in eval_items:
        legal = None if value_only else leaf.legal_moves
        moves_list.append(
            torch.stack([encode_move(m) for m in legal]).to(device) if legal else None
        )
    with torch.no_grad():
        values, priors_list = model.batch_eval(positions, moves_list)
    vals = values.tolist()
    for j, (path, leaf, value_only) in enumerate(eval_items):
        if not value_only and leaf.legal_moves and not leaf.expanded:
            _expand_with_priors(leaf, leaf.legal_moves, priors_list[j].tolist(), client)
        _backup(path, vals[j])


def _make_root(state: GameState, client) -> PuctNode:
    legal = state.get("legalMoves") or []
    root = PuctNode(fen=state["fen"], move_to_here=None, legal_moves=legal)
    if hasattr(client, "parse"):
        try:
            root.state = client.parse(state["fen"])
        except Exception:  # noqa: BLE001
            pass
    return root


def run_mcts_vectorized(
    states: list[GameState],
    client,
    model: ChesskersScorer,
    n_sims: int,
    c_puct: float = 1.5,
    device: torch.device | None = None,
    dirichlet_alpha: float | None = None,
    dirichlet_eps: float = 0.25,
) -> list[MctsResult]:
    """Run n_sims of PUCT for each of K games (states) in lockstep. Returns one
    MctsResult per game. One batch_eval per macro-step (batch size = #games with
    a non-terminal leaf this step)."""
    if device is None:
        device = next(model.parameters()).device
    roots = [_make_root(st, client) for st in states]

    # Macro-step 0: expand every root in a single batch.
    root_items = [([r], r, False) for r in roots if r.legal_moves]
    _eval_expand_backup(root_items, client, model, device)
    if dirichlet_alpha is not None:
        for r in roots:
            if r.children:
                _apply_dirichlet_noise(r, dirichlet_alpha, dirichlet_eps)

    for _ in range(max(0, n_sims - 1)):
        eval_items = []
        for root in roots:
            if not root.children:
                continue
            path, leaf = _select_to_leaf(root, c_puct)
            if leaf.is_terminal:
                _backup(path, _terminal_value(leaf))
            elif leaf.expanded:  # expanded, no children — value-only fallback
                eval_items.append((path, leaf, True))
            else:
                eval_items.append((path, leaf, False))
        _eval_expand_backup(eval_items, client, model, device)

    results = []
    for root in roots:
        if not root.children:
            results.append(MctsResult(chosen=None, visit_distribution={}, root=root))
        else:
            vd = {uci: c.visits for uci, c in root.children.items()}
            best = max(root.children.values(), key=lambda c: c.visits)
            results.append(MctsResult(chosen=best.move_to_here, visit_distribution=vd, root=root))
    return results


# ---------------------------------------------------------------------------
# Benchmark + equivalence harness
# ---------------------------------------------------------------------------

SEEDS = [
    "7p/8/8/8/8/8/8/4K3[h8:ssss] b - - 0 1",
    "8/8/3kkk2/8/8/8/3PPP2/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
    "8/8/8/8/3kk3/8/8/4K3[d4:kk,e4:kk] b - - 0 1",
    "3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] b - - 0 1",
]


def verify_equiv(model, device) -> None:
    """K=1, no noise: vectorized visit distribution must equal run_mcts exactly."""
    from chessckers_engine.mcts_puct import run_mcts
    client = PyVariantClient()
    ok = True
    for fen in SEEDS:
        st = client.new_game(fen)
        seq = run_mcts(st, client, model, n_sims=64, c_puct=1.5).visit_distribution
        vec = run_mcts_vectorized([st], client, model, n_sims=64, c_puct=1.5, device=device)[0].visit_distribution
        match = seq == vec
        ok = ok and match
        print(f"  {'OK ' if match else 'MISMATCH'} {fen[:34]:34}  seq_top={max(seq.values()) if seq else 0} vec_top={max(vec.values()) if vec else 0}")
    print(f"equivalence: {'PASS' if ok else 'FAIL'}\n")


def bench_vectorized(model, device, K, sims, n_moves, rng) -> dict:
    """Keep K games in flight (refill from SEEDS on game end); play n_moves total
    moves (argmax). Report games/sec-equivalent via moves/sec, evals (= batches),
    and positions evaluated/sec."""
    client = PyVariantClient()
    states = [client.new_game(rng.choice(SEEDS)) for _ in range(K)]
    moves_played = 0
    pos_evaluated = 0
    t0 = time.perf_counter()
    while moves_played < n_moves:
        results = run_mcts_vectorized(states, client, model, n_sims=sims, c_puct=1.5, device=device)
        pos_evaluated += sum(1 for r in results if r.root.children) * sims  # ~ per-game evals
        for i, r in enumerate(results):
            if r.chosen is None:
                states[i] = client.new_game(rng.choice(SEEDS))
                continue
            nxt = client.make_move(states[i]["fen"], r.chosen["uci"])
            moves_played += 1
            if nxt.get("status") or not (nxt.get("legalMoves") or []):
                states[i] = client.new_game(rng.choice(SEEDS))
            else:
                states[i] = nxt
    elapsed = time.perf_counter() - t0
    return {
        "moves": moves_played, "secs": elapsed,
        "moves_per_sec": moves_played / elapsed,
        "pos_per_sec": pos_evaluated / elapsed,
    }


def bench_cpu_seq(model, sims, n_moves, rng) -> dict:
    """Baseline: one game at a time, per-game run_mcts (batch-1). The production
    per-worker primitive; aggregate = this x n_processes."""
    from chessckers_engine.mcts_puct import run_mcts
    client = PyVariantClient()
    state = client.new_game(rng.choice(SEEDS))
    moves_played = 0
    pos_evaluated = 0
    t0 = time.perf_counter()
    while moves_played < n_moves:
        r = run_mcts(state, client, model, n_sims=sims, c_puct=1.5)
        if r.root.children:
            pos_evaluated += sims
        if r.chosen is None:
            state = client.new_game(rng.choice(SEEDS)); continue
        nxt = client.make_move(state["fen"], r.chosen["uci"])
        moves_played += 1
        state = nxt if not (nxt.get("status") or not (nxt.get("legalMoves") or [])) else client.new_game(rng.choice(SEEDS))
    elapsed = time.perf_counter() - t0
    return {"moves": moves_played, "secs": elapsed, "moves_per_sec": moves_played / elapsed,
            "pos_per_sec": pos_evaluated / elapsed}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["verify", "cpu-seq", "mps-vec", "cpu-vec"], default="verify")
    p.add_argument("--device", default="")  # override; default per mode
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--moves", type=int, default=400, help="total moves to play")
    p.add_argument("--weights", default="")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device_name = args.device or ("mps" if args.mode == "mps-vec" else "cpu")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS not available")
    device = torch.device(device_name)
    rng = random.Random(args.seed)

    model = ChesskersScorer(d_hidden=256, c_filters=96, n_blocks=4).to(device).eval()
    if args.weights:
        from chessckers_engine.checkpoints import load_checkpoint
        load_checkpoint(model, args.weights)

    if args.mode == "verify":
        print(f"=== equivalence vectorized(K=1) vs run_mcts (device={device_name}) ===")
        verify_equiv(model, device)
        return 0

    # warmup
    if args.mode in ("mps-vec", "cpu-vec"):
        run_mcts_vectorized([PyVariantClient().new_game(SEEDS[0])] * min(args.K, 8),
                            PyVariantClient(), model, n_sims=20, device=device)
    else:
        from chessckers_engine.mcts_puct import run_mcts
        run_mcts(PyVariantClient().new_game(SEEDS[0]), PyVariantClient(), model, n_sims=20)

    if args.mode == "cpu-seq":
        r = bench_cpu_seq(model, args.sims, args.moves, rng)
        print(f"=== cpu-seq (batch-1, threads={torch.get_num_threads()}, sims={args.sims}) ===")
    else:
        r = bench_vectorized(model, device, args.K, args.sims, args.moves, rng)
        print(f"=== {args.mode} (K={args.K}, device={device_name}, threads={torch.get_num_threads()}, sims={args.sims}) ===")
    print(f"moves={r['moves']}  wall={r['secs']:.2f}s  "
          f"moves/sec={r['moves_per_sec']:.2f}  pos_eval/sec={r['pos_per_sec']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
