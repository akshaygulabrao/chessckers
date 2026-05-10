"""Show the network's policy + value output on a list of positions.

Usage:
    python inspect_network.py                          # default position bank
    python inspect_network.py --weights PATH           # specific weights file
    python inspect_network.py --fens FILE              # custom FENs (one per line)
    python inspect_network.py --top-k 8 --mcts-sims 0  # tweak display / add MCTS

Each position prints:
    - ASCII board (with side-to-move marker)
    - Value head output (in [-1, 1], from side-to-move's POV)
    - Top-K legal moves with policy priors and (if --mcts-sims > 0) MCTS visit %

Default weights = `runs/local-004/weights.pt`. Override with --weights.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

DEFAULT_WEIGHTS = Path(__file__).parent / "runs" / "local-004" / "weights.pt"

# A small bank of positions worth inspecting. Each entry is (label, fen).
DEFAULT_POSITIONS = [
    (
        "starting position (white to move)",
        # Canonical Chessckers starting FEN, with the stack overlay.
        None,  # filled in below from variant_py.STARTING_FEN
    ),
    (
        "after 1.e4 (black to move — should pick a black stack move)",
        None,  # filled at runtime from PyVariantClient
    ),
    (
        "after 1.e4 + black plays a6-stack one square diagonally (white to move)",
        None,  # filled at runtime
    ),
]


def _print_board(fen: str, label: str) -> None:
    """Render a Chessckers FEN as a board grid. The bracketed `[a6:s,...]`
    suffix is stack-overlay metadata that python-chess can't parse, so we
    splice it out while keeping the side-to-move and castling fields."""
    import chess
    # FEN looks like "<board>[<overlay>] <stm> <castling> <ep> <halfmove> <fullmove>".
    # Strip the bracketed overlay but keep all space-separated fields after it.
    pre, _, rest = fen.partition("[")
    if rest:
        _, _, after_bracket = rest.partition("]")
        base_fen = (pre + after_bracket).strip()
    else:
        base_fen = fen
    board = chess.Board(base_fen)
    print(f"--- {label} ---")
    print(f"FEN:  {fen}")
    print(board.unicode(borders=False, empty_square="·", invert_color=False))
    print(f"side to move: {'white' if board.turn else 'black'}")


def _format_move(m: dict) -> str:
    """Compact one-line move description: UCI plus tags (capture/chain/etc)."""
    uci = m.get("uci", "?")
    tags = []
    if m.get("isCapture"):
        tags.append("x")
    if m.get("waypoints"):
        tags.append(f"chain[{len(m['waypoints'])}]")
    if m.get("deployCount"):
        tags.append(f"deploy{m['deployCount']}")
    if m.get("demotionsRequired"):
        tags.append(f"demote{m['demotionsRequired']}")
    suffix = (" " + ",".join(tags)) if tags else ""
    return f"{uci}{suffix}"


def _legal_moves(state, client) -> list[dict]:
    """Return the legal-move list for `state` via `status_and_legal`.
    Returns [] if the position is terminal."""
    status, _winner, moves = client.status_and_legal(state)
    if status is not None or moves is None:
        return []
    return list(moves)


def _eval_position(model, fen: str, top_k: int, mcts_sims: int, device: str) -> None:
    """Print the network's view of `fen`: value, top-K policy priors,
    and (if requested) MCTS visit distribution."""
    from chessckers_engine.encoding import encode_move, encode_position
    from chessckers_engine.variant_py import PyVariantClient
    from chessckers_engine.variant_py.state import parse_fen

    client = PyVariantClient()
    state = parse_fen(fen)
    moves = _legal_moves(state, client)
    if not moves:
        print("(terminal position — no legal moves)")
        return

    pos_t = encode_position(fen).unsqueeze(0).to(device)
    move_t = torch.stack([encode_move(m) for m in moves]).to(device)
    with torch.no_grad():
        logits, value = model.policy_and_value(pos_t, move_t)
        priors = torch.softmax(logits, dim=0).cpu().tolist()
        v = float(value.cpu())

    # Sort by prior descending, take top K.
    rows = sorted(zip(moves, priors), key=lambda r: r[1], reverse=True)[:top_k]
    print(f"value head: {v:+.3f}  (range [-1, 1], from STM POV)")
    print(f"legal moves: {len(moves)}  (showing top {len(rows)} by prior)")

    visits = None
    if mcts_sims > 0:
        from chessckers_engine.mcts_puct import PuctMcts
        mcts = PuctMcts(model, c_puct=1.5, dirichlet_alpha=None,
                        dirichlet_eps=0.0, vloss_batch=1)
        result = mcts.run(state, n_sims=mcts_sims, client=client)
        visits = {m["uci"]: c for m, c in zip(result.moves, result.visit_counts)}
        total_visits = sum(visits.values()) or 1

    for m, p in rows:
        line = f"  {p*100:5.1f}%  {_format_move(m)}"
        if visits is not None:
            v_count = visits.get(m["uci"], 0)
            line += f"   mcts: {v_count:>4}  ({v_count / total_visits * 100:5.1f}%)"
        print(line)
    print()


def _resolve_default_positions() -> list[tuple[str, str]]:
    """Build the default position bank: starting position + a couple variations."""
    from chessckers_engine.variant_py import PyVariantClient
    from chessckers_engine.variant_py.state import STARTING_FEN

    client = PyVariantClient()
    # Position 1: starting FEN
    out = [("starting position (white to move)", STARTING_FEN)]

    # Position 2: after 1.e2-e4
    s = client.new_game()
    s = client.make_move(s["fen"], "e2e4")
    out.append(("after 1. e4 (black to move)", s["fen"]))

    # Position 3: after black plays its top-prior move (whatever that is)
    # — picked deterministically from the move list to stay reproducible.
    moves = s["legalMoves"]
    if moves:
        # Prefer a non-capture stack slide so the position stays "early game".
        def _key(m):
            return (m.get("isCapture", False), m.get("uci", ""))
        moves_sorted = sorted(moves, key=_key)
        first_move = moves_sorted[0]
        s = client.make_move(s["fen"], first_move["uci"])
        out.append((f"after 1. e4 {first_move['uci']} (white to move)", s["fen"]))

    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                   help="path to weights.pt (default: runs/local-004/weights.pt)")
    p.add_argument("--fens", type=Path, default=None,
                   help="file of FENs to inspect, one per line; optional `# comment` after")
    p.add_argument("--top-k", type=int, default=8,
                   help="how many top-prior moves to show per position (default 8)")
    p.add_argument("--mcts-sims", type=int, default=0,
                   help="if >0, also run MCTS for N sims and show visit counts")
    p.add_argument("--device", default="cpu",
                   help="torch device for the model (default cpu)")
    p.add_argument("--d-hidden", type=int, default=256)
    p.add_argument("--c-filters", type=int, default=128)
    p.add_argument("--n-blocks", type=int, default=6)
    args = p.parse_args()

    if not args.weights.exists():
        print(f"weights not found: {args.weights}", file=sys.stderr)
        return 2

    from chessckers_engine.checkpoints import load_checkpoint
    from chessckers_engine.model import ChesskersScorer

    device = torch.device(args.device)
    model = ChesskersScorer(d_hidden=args.d_hidden, c_filters=args.c_filters,
                            n_blocks=args.n_blocks).to(device).eval()
    load_checkpoint(model, args.weights)
    print(f"loaded weights: {args.weights} ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params, {device})\n")

    if args.fens is not None:
        positions = []
        for raw in args.fens.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            positions.append((args.fens.name, line))
    else:
        positions = _resolve_default_positions()

    for label, fen in positions:
        _print_board(fen, label)
        _eval_position(model, fen, args.top_k, args.mcts_sims, str(device))
    return 0


if __name__ == "__main__":
    sys.exit(main())
