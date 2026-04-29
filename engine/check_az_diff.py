"""Diagnostic: load each iter-az checkpoint and print (value, top priors) on
two positions. If outputs are identical across iters, the network isn't being
trained or has collapsed (uniform softmax / saturated tanh).

Run from engine/:
  uv run python check_az_diff.py                # default: weights/
  uv run python check_az_diff.py weights/ln     # other dir
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from chessckers_engine.checkpoints import load_checkpoint
from chessckers_engine.encoding import encode_move, encode_position
from chessckers_engine.model import ChesskersScorer

ENGINE = Path(__file__).resolve().parent

START_FEN = (
    "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"
    "[a6:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,"
    "a7:k,b7:k,c7:k,d7:k,e7:k,f7:k,g7:k,h7:k,"
    "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] w KQkq - 0 1"
)


def m(fr: str, to: str, **kw) -> dict:
    out = {"from": fr, "to": to}
    out.update(kw)
    return out


START_LEGAL = (
    [m(f"{c}2", f"{c}3") for c in "abcdefgh"]
    + [m(f"{c}2", f"{c}4") for c in "abcdefgh"]
    + [m("b1", "a3"), m("b1", "c3"), m("g1", "f3"), m("g1", "h3")]
)

# A mid-game position from a real saved game (move ~10), with hand-picked legals
# spanning quiet pawn moves, a piece move, and a chain-capture for variety.
MID_FEN = (
    "pppppppp/kkkk1kkk/pppkpppp/8/8/PP6/2PPPPPP/RNBQKBNR"
    "[a6:s,b6:s,c6:s,d6:sk,e6:s,f6:s,g6:s,h6:s,"
    "a7:k,b7:k,c7:k,d7:k,f7:k,g7:k,h7:k,"
    "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] b KQ - 0 1"
)
# Hand-curated Black moves from this position (Black to move, lots of stones on r6).
MID_LEGAL = [
    m("a8", "b7"),
    m("b8", "a7"),
    m("c8", "b7"),
    m("e8", "f7"),
    m("f8", "g7"),
    m("h8", "g7"),
    m("d6", "c5"),  # diag descent (king-on-stone stack)
    m("d6", "e5"),
    m("a6", "b5"),
    m("h6", "g5"),
    m("d7", "e6"),  # stack onto e6
    m("g7", "f6"),
]


def score(ckpt: Path, fen: str, legal: list[dict]) -> tuple[float, list[tuple[str, float]]]:
    model = ChesskersScorer()
    load_checkpoint(model, ckpt)
    model.eval()
    pos = encode_position(fen).unsqueeze(0)
    moves = torch.stack([encode_move(mv) for mv in legal])
    with torch.no_grad():
        logits, v = model.policy_and_value(pos, moves)
    probs = torch.softmax(logits, dim=0).tolist()
    pairs = sorted(
        zip([f"{mv['from']}{mv['to']}" for mv in legal], probs),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return float(v.item()), pairs[:5]


def main() -> int:
    weights = Path(sys.argv[1]) if len(sys.argv) > 1 else ENGINE / "weights"
    if not weights.is_absolute():
        weights = ENGINE / weights
    ckpts = sorted(weights.glob("iter-az-*.pt"))
    if not ckpts:
        print(f"no iter-az-*.pt checkpoints found in {weights}")
        return 1
    n_start, n_mid = len(START_LEGAL), len(MID_LEGAL)
    print(f"comparing {len(ckpts)} checkpoints in {weights} on two positions\n")
    print(f"START (white to move, {n_start} legals; uniform = {1/n_start:.3f}):")
    for c in ckpts:
        v, top = score(c, START_FEN, START_LEGAL)
        top_str = ", ".join(f"{u}={p:.3f}" for u, p in top)
        print(f"  {c.name}  v={v:+.4f}  top5: {top_str}")
    print(f"\nMID-GAME (black to move, {n_mid} legals; uniform = {1/n_mid:.3f}):")
    for c in ckpts:
        v, top = score(c, MID_FEN, MID_LEGAL)
        top_str = ", ".join(f"{u}={p:.3f}" for u, p in top)
        print(f"  {c.name}  v={v:+.4f}  top5: {top_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
