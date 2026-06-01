"""Compare several networks' output on a single position (default: the start).

For each checkpoint, prints the value-head output and the top-K legal moves by
policy prior. The legal-move list itself comes from the move generator (not the
network) — the net only scores each candidate (see encode_move / model.head).

Usage:
    python compare_networks.py WEIGHTS...            # one or more .pt files
    python compare_networks.py weights/iter-az-*.pt  # glob-expanded by the shell
    python compare_networks.py --fen '<FEN>' W...    # a different position
    python compare_networks.py --top-k 8 W...

All checkpoints must share the architecture given by --model-* (defaults match
the self-play loop: d_hidden=256, c_filters=96, n_blocks=4). A checkpoint with
a different shape fails loudly rather than loading garbage at random init.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _print_board(fen: str, label: str) -> None:
    """Render a Chessckers FEN board. The bracketed `[a6:s,...]` stack overlay
    isn't python-chess-parseable, so splice it out but keep the trailing fields."""
    import chess
    pre, _, rest = fen.partition("[")
    base_fen = (pre + rest.partition("]")[2]).strip() if rest else fen
    board = chess.Board(base_fen)
    print(f"position: {label}")
    print(f"FEN: {fen}")
    print(board.unicode(borders=False, empty_square="·"))
    print(f"side to move: {'white' if board.turn else 'black'}\n")


def _format_move(m: dict) -> str:
    """Compact move label: UCI plus capture/chain/deploy/demote tags."""
    tags = []
    if m.get("capture") is not None:
        tags.append("x")
    if m.get("waypoints"):
        tags.append(f"chain[{len(m['waypoints'])}]")
    if m.get("deployCount") is not None:
        tags.append(f"deploy{m['deployCount']}")
    if m.get("demotionsRequired") is not None:
        tags.append(f"demote{m['demotionsRequired']}")
    suffix = (" " + ",".join(tags)) if tags else ""
    return f"{m.get('uci', '?')}{suffix}"


def _eval_one(model, fen: str, moves: list[dict], top_k: int, device: str) -> None:
    """Print one model's value + top-K policy priors for `fen`."""
    from chessckers_engine.encoding import encode_move, encode_position

    pos_t = encode_position(fen).unsqueeze(0).to(device)
    move_t = torch.stack([encode_move(m) for m in moves]).to(device)
    with torch.no_grad():
        logits, value = model.policy_and_value(pos_t, move_t)
        priors = torch.softmax(logits, dim=0).cpu().tolist()
        v = float(value.cpu())

    rows = sorted(zip(moves, priors), key=lambda r: r[1], reverse=True)[:top_k]
    print(f"  value: {v:+.3f}   (STM POV, range [-1, 1])")
    print(f"  {len(moves)} legal moves — top {len(rows)} by prior:")
    for m, p in rows:
        print(f"    {p * 100:5.1f}%  {_format_move(m)}")
    print()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", nargs="+", type=Path, help="checkpoint .pt file(s)")
    p.add_argument("--fen", default=None, help="position FEN (default: starting position)")
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--device", default="cpu")
    p.add_argument("--model-hidden", type=int, default=256)
    p.add_argument("--model-filters", type=int, default=96)
    p.add_argument("--model-blocks", type=int, default=4)
    args = p.parse_args()

    from chessckers_engine.checkpoints import load_checkpoint
    from chessckers_engine.model import ChesskersScorer
    from chessckers_engine.variant_py import PyVariantClient
    from chessckers_engine.variant_py.state import STARTING_FEN, parse_fen

    fen = args.fen or STARTING_FEN
    client = PyVariantClient()
    state = parse_fen(fen)
    status, _winner, moves = client.status_and_legal(state)
    if status is not None or not moves:
        print(f"position is terminal ({status}) — no legal moves to score", file=sys.stderr)
        return 2
    moves = list(moves)

    _print_board(fen, "starting position" if args.fen is None else "custom")

    device = torch.device(args.device)
    for w in args.weights:
        if not w.exists():
            print(f"=== {w} ===\n  (not found)\n", file=sys.stderr)
            continue
        model = ChesskersScorer(d_hidden=args.model_hidden, c_filters=args.model_filters,
                                n_blocks=args.model_blocks).to(device).eval()
        try:
            load_checkpoint(model, w)
        except RuntimeError as e:
            print(f"=== {w} ===\n  FAILED to load (arch mismatch?): {str(e)[:100]}\n",
                  file=sys.stderr)
            continue
        n_params = sum(t.numel() for t in model.parameters()) / 1e6
        print(f"=== {w}  ({n_params:.2f}M params) ===")
        _eval_one(model, fen, moves, args.top_k, str(device))
    return 0


if __name__ == "__main__":
    sys.exit(main())
