"""ChesskersScorer: a (position, candidate moves) → per-move logit network,
with an additional scalar value head for position evaluation.

Architecture:
- Position trunk (shared between heads): 2× Conv2d (c_in → 32 → 64) with 3x3
  kernels and padding=1, ReLU between, flatten + Linear → d_hidden + ReLU.
- Move encoder: Linear(d_move → d_hidden) + ReLU.
- Policy head (`head` for backward compat with M4-phase-1 checkpoints):
  concat(pos_emb broadcast over N moves, move_emb) → Linear(2*d_hidden →
  d_hidden) + ReLU + Linear(d_hidden → 1) → squeeze.
- Value head: Linear(d_hidden → d_hidden//2) + ReLU + Linear(d_hidden//2 → 1)
  + Tanh. Output in [-1, 1] matches AlphaZero-style outcome targets
  (+1 win / 0 draw / -1 loss) from the side-to-move's perspective.

`forward(position, moves)` keeps the M3 contract — returns per-move logits.
`value(position)` is the new method MCTS leaves use.

The value head is randomly initialized. Pre-AlphaZero checkpoints (no
value_head keys) load fine via `checkpoints.load_checkpoint` which uses
strict=False; the value head stays at random init until self-play training
fills it in.
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
        # Policy head; named `head` for backward compat with M4-phase-1 checkpoints.
        self.head = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden // 2, 1),
            nn.Tanh(),
        )

    def _position_embedding(self, position: torch.Tensor) -> torch.Tensor:
        if position.dim() != 4 or position.shape[0] != 1:
            raise ValueError(f"position must be (1, C, 8, 8); got {tuple(position.shape)}")
        return self.position_trunk(position)  # (1, d_hidden)

    def forward(self, position: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
        """
        position: (1, C, 8, 8) — single position
        moves:    (N, D)       — N candidate move feature vectors
        returns:  (N,)         — one policy logit per candidate
        """
        if moves.dim() != 2:
            raise ValueError(f"moves must be (N, D); got {tuple(moves.shape)}")
        n = moves.shape[0]
        pos_emb = self._position_embedding(position)        # (1, d_hidden)
        pos_emb = pos_emb.expand(n, -1)                     # (N, d_hidden)
        move_emb = self.move_encoder(moves)                 # (N, d_hidden)
        combined = torch.cat([pos_emb, move_emb], dim=1)    # (N, 2*d_hidden)
        return self.head(combined).squeeze(-1)              # (N,)

    def value(self, position: torch.Tensor) -> torch.Tensor:
        """Scalar position value from the side-to-move's perspective, in [-1, 1].

        position: (1, C, 8, 8)
        returns:  () scalar tensor
        """
        emb = self._position_embedding(position)            # (1, d_hidden)
        return self.value_head(emb).reshape(())             # ()

    def policy_and_value(
        self, position: torch.Tensor, moves: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One forward over the position trunk producing both heads' outputs."""
        if moves.dim() != 2:
            raise ValueError(f"moves must be (N, D); got {tuple(moves.shape)}")
        n = moves.shape[0]
        pos_emb = self._position_embedding(position)        # (1, d_hidden)
        v = self.value_head(pos_emb).reshape(())            # ()
        pos_emb_n = pos_emb.expand(n, -1)
        move_emb = self.move_encoder(moves)
        combined = torch.cat([pos_emb_n, move_emb], dim=1)
        logits = self.head(combined).squeeze(-1)            # (N,)
        return logits, v
