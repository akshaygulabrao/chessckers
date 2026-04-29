"""Probe the policy-head pre/post-ReLU activations and final logits across
checkpoints. Flags both dead-ReLU collapse (post-ReLU all zeros) and runaway
explosion (logits so large that float32 rounds them to identical values).

Run from engine/:
  uv run python check_dead_relu.py             # default: weights/
  uv run python check_dead_relu.py weights/ln  # other dir
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
LEGAL = (
    [{"from": f"{c}2", "to": f"{c}3"} for c in "abcdefgh"]
    + [{"from": f"{c}2", "to": f"{c}4"} for c in "abcdefgh"]
    + [
        {"from": "b1", "to": "a3"},
        {"from": "b1", "to": "c3"},
        {"from": "g1", "to": "f3"},
        {"from": "g1", "to": "h3"},
    ]
)


def probe(ckpt: Path) -> None:
    model = ChesskersScorer()
    load_checkpoint(model, ckpt)
    model.eval()
    pos = encode_position(START_FEN).unsqueeze(0)
    moves = torch.stack([encode_move(mv) for mv in LEGAL])
    with torch.no_grad():
        pos_emb = model._position_embedding(pos)
        pos_emb_n = pos_emb.expand(moves.shape[0], -1)
        move_emb = model.move_encoder(moves)
        combined = torch.cat([pos_emb_n, move_emb], dim=1)
        # Locate the (last) ReLU and the (last) Linear inside head, regardless
        # of whether LayerNorm has been inserted between them.
        layers = list(model.head)
        relu_idx = max(i for i, m in enumerate(layers) if isinstance(m, torch.nn.ReLU))
        last_linear = layers[-1]
        x = combined
        for layer in layers[:relu_idx]:
            x = layer(x)
        pre = x  # pre-ReLU activation
        post = layers[relu_idx](pre)
        logits = last_linear(post).squeeze(-1)
    print(f"\n{ckpt.name}:")
    print(f"  pre-ReLU stats:  min={pre.min():.3f}  max={pre.max():.3f}  "
          f"mean={pre.mean():.3f}  frac>0={(pre > 0).float().mean():.3f}")
    print(f"  post-ReLU stats: min={post.min():.3f}  max={post.max():.3f}  "
          f"mean={post.mean():.3f}  nonzero_units_per_move={(post > 0).any(dim=0).sum()}/{post.shape[1]}")
    print(f"  logits range: [{logits.min():.4f}, {logits.max():.4f}]  "
          f"spread={logits.max() - logits.min():.6f}")
    print(f"  last-Linear bias: {last_linear.bias.item():.4f}")


weights = Path(sys.argv[1]) if len(sys.argv) > 1 else ENGINE / "weights"
if not weights.is_absolute():
    weights = ENGINE / weights
ckpts = sorted(weights.glob("iter-az-*.pt"))
if not ckpts:
    print(f"no iter-az-*.pt in {weights}")
    sys.exit(1)
print(f"probing {len(ckpts)} checkpoints in {weights}")
for c in ckpts:
    probe(c)
