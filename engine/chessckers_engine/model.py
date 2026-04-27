"""ChesskersScorer: a (position, candidate moves) → per-move logit network.

Architecture:
- Position trunk: 2× Conv2d (c_in → 32 → 64) with 3x3 kernels and padding=1,
  ReLU between, then flatten + Linear → d_hidden.
- Move encoder: Linear(d_move → d_hidden) + ReLU.
- Combine head: concat(pos_emb broadcast over N moves, move_emb) → Linear(2*d_hidden → d_hidden) + ReLU + Linear(d_hidden → 1) → squeeze.

Forward consumes a single position (batch dim 1) and a stack of N candidate
moves; returns a (N,) tensor of logits ready for argmax or softmax.

Random initialization is sufficient for milestone 3 (scaffolding). Training
will land in a later milestone.
"""

from __future__ import annotations

import torch
from torch import nn

from chessckers_engine.encoding import MOVE_D, POS_C


class ChesskersScorer(nn.Module):
    def __init__(self, c_in: int = POS_C, d_move: int = MOVE_D, d_hidden: int = 128):
        super().__init__()
        self.position_trunk = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, d_hidden),
            nn.ReLU(inplace=True),
        )
        self.move_encoder = nn.Sequential(
            nn.Linear(d_move, d_hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, position: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
        """
        position: (1, C, 8, 8) — single position
        moves:    (N, D)       — N candidate move feature vectors
        returns:  (N,)         — one logit per candidate
        """
        if position.dim() != 4 or position.shape[0] != 1:
            raise ValueError(f"position must be (1, C, 8, 8); got {tuple(position.shape)}")
        if moves.dim() != 2:
            raise ValueError(f"moves must be (N, D); got {tuple(moves.shape)}")
        n = moves.shape[0]
        pos_emb = self.position_trunk(position)            # (1, d_hidden)
        pos_emb = pos_emb.expand(n, -1)                    # (N, d_hidden)
        move_emb = self.move_encoder(moves)                # (N, d_hidden)
        combined = torch.cat([pos_emb, move_emb], dim=1)   # (N, 2*d_hidden)
        return self.head(combined).squeeze(-1)             # (N,)
