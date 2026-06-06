"""ChesskersScorer: a (position, candidate moves) → per-move logit network,
with an additional scalar value head for position evaluation.

Architecture (residual-tower variant, ~2.4M params at default settings):
- Initial conv: Conv2d(c_in → c_filters) + GroupNorm + ReLU.
- Residual tower: `n_blocks` × ResidualBlock(c_filters)
    each block = Conv → GN → ReLU → Conv → GN → (+input) → ReLU.

GroupNorm is used instead of BatchNorm because MCTS does single-position
forward calls (batch=1). BatchNorm with batch=1 in train mode computes
variance over a single sample (= 0) → division by ε → unstable. GroupNorm
is invariant to batch size and avoids the train/eval-mode footgun entirely.
- Bottleneck: Flatten + Linear(c_filters*64 → d_hidden) + LayerNorm + ReLU.
- Move encoder: Linear(d_move → d_hidden) + LayerNorm + ReLU.
- Policy head (`head` for backward compat with M4-phase-1 checkpoints):
  concat(pos_emb broadcast over N moves, move_emb) → Linear(2*d_hidden →
  d_hidden) + LayerNorm + ReLU + Linear(d_hidden → 1) → squeeze.
- Value head: Linear(d_hidden → d_hidden//2) + LayerNorm + ReLU +
  Linear(d_hidden//2 → 1) + Tanh.

The residual structure replaces the previous flat 2-conv trunk. LayerNorm
after every hidden Linear (and BatchNorm inside conv blocks) holds
activations bounded — an earlier training run without normalization blew
policy logits to ~3e7 by iter-2, collapsing the softmax to uniform.
See `engine/check_dead_relu.py` for the post-mortem.

`forward(position, moves)` returns per-move logits.
`value(position)` is what MCTS leaves use.

Old checkpoints (different param shapes) won't load against this model;
the residual-tower refactor is a fresh-init checkpoint compatibility break.
`checkpoints.load_checkpoint` uses strict=False, so loading an old
checkpoint will keep the new params at random init and log the missing
keys, but the resulting model is effectively un-trained.
"""

from __future__ import annotations

import torch
from torch import nn

from chessckers_engine.encoding import (
    MOVE_D,
    MOVE_D_V2,
    MV2_PATH_BASE,
    MV2_SCALAR_BASE,
    POS_C,
    POS_C_V2,
)


class ResidualBlock(nn.Module):
    """Pre-activation residual block: Conv → BN → ReLU → Conv → BN → (+x) → ReLU."""

    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=c)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups=8, num_channels=c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)


class ChesskersScorer(nn.Module):
    def __init__(
        self,
        c_in: int = POS_C,
        d_move: int = MOVE_D,
        d_hidden: int = 256,
        c_filters: int = 96,
        n_blocks: int = 4,
    ):
        super().__init__()
        # Position trunk: initial conv + residual tower + bottleneck dense.
        # Kept as a single Sequential so existing tests that walk the trunk
        # looking for Conv2d modules still find the input conv at index 0.
        self.position_trunk = nn.Sequential(
            nn.Conv2d(c_in, c_filters, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=c_filters),
            nn.ReLU(inplace=True),
            *[ResidualBlock(c_filters) for _ in range(n_blocks)],
            nn.Flatten(),
            nn.Linear(c_filters * 8 * 8, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(inplace=True),
        )
        self.move_encoder = nn.Sequential(
            nn.Linear(d_move, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(inplace=True),
        )
        # Policy head; named `head` for backward compat with M4-phase-1 checkpoints.
        self.head = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, 1),
        )
        # WDL value head: 3 logits (Win/Draw/Loss from the side-to-move's POV).
        # MCTS uses the scalar Q = P(win) - P(loss) (see value()/batch_eval).
        # Replaces the old scalar-tanh head — better calibration + draw-awareness
        # (Lc0-style); the moves-left head below handles win-speed.
        self.value_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden // 2, 3),
        )
        # Moves-left head: plies-to-end estimate (>=0 via Softplus). Trains the
        # net to prefer faster wins / slower losses — the principled shortest-mate
        # fix (vs the old value-discount hack).
        self.moves_left_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden // 2, 1),
            nn.Softplus(),
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
        wdl = torch.softmax(self.value_head(emb), dim=-1)   # (1, 3)
        return (wdl[..., 0] - wdl[..., 2]).reshape(())      # Q = P(win) - P(loss)

    def batch_eval(
        self, positions: torch.Tensor, moves_list: list[torch.Tensor | None]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Evaluate B positions and their (ragged) move lists in one batched
        pass — both the trunk+value-head AND the policy head are batched. The
        ragged move lists are padded to the longest (B, n_max, D), the policy
        head (row-wise: Linear/LayerNorm/ReLU over the feature dim only) runs
        over all B*n_max slots at once, and the padded logits are masked to
        -inf before the per-position softmax. Because the head is row-wise,
        padding never contaminates real moves, so each position's prior
        distribution is bit-for-bit identical to the per-position computation.

        positions:  (B, C, 8, 8)
        moves_list: list of B items; each is a (N_i, D) tensor or None/(0, D)
                    for "no moves" (terminal-ish or value-only requests).
        Returns:    (values: (B,), priors_list: list of B (N_i,) tensors).
                    For positions with no moves, the corresponding priors entry
                    is an empty tensor."""
        if positions.dim() != 4:
            raise ValueError(f"positions must be (B, C, 8, 8); got {tuple(positions.shape)}")
        if positions.shape[0] != len(moves_list):
            raise ValueError(
                f"batch size mismatch: positions B={positions.shape[0]}, "
                f"moves_list len={len(moves_list)}"
            )
        # One trunk pass for all positions (the expensive part).
        pos_emb = self.position_trunk(positions)             # (B, d_hidden)
        wdl = torch.softmax(self.value_head(pos_emb), dim=-1)  # (B, 3)
        values = wdl[:, 0] - wdl[:, 2]                       # (B,) scalar Q = P(win)-P(loss)

        B = positions.shape[0]
        device = positions.device
        lengths = [
            0 if (m is None or m.shape[0] == 0) else int(m.shape[0])
            for m in moves_list
        ]
        n_max = max(lengths, default=0)
        if n_max == 0:
            # Value-only batch: no candidate moves anywhere.
            return values, [torch.empty(0, device=device) for _ in range(B)]

        d_move = next(
            m.shape[1] for m in moves_list if m is not None and m.shape[0] > 0
        )
        # Pad ragged move lists to (B, n_max, D) + a validity mask.
        padded = positions.new_zeros((B, n_max, d_move))
        mask = torch.zeros((B, n_max), dtype=torch.bool, device=device)
        for i, moves in enumerate(moves_list):
            if lengths[i]:
                padded[i, : lengths[i]] = moves
                mask[i, : lengths[i]] = True

        logits = self._batched_move_logits(pos_emb, padded, mask)
        priors = torch.softmax(logits, dim=1)                # (B, n_max); padding → 0
        # Rows with no moves softmax to NaN (all -inf) but are sliced to empty
        # below and never read.
        return values, [
            priors[i, : lengths[i]] if lengths[i] else torch.empty(0, device=device)
            for i in range(B)
        ]

    def _batched_move_logits(
        self, pos_emb: torch.Tensor, padded_moves: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Policy-head logits for a padded batch of ragged move lists — shared by
        batch_eval (MCTS priors) and the training loss so the two can't diverge.

        pos_emb:      (B, d_hidden) — one position embedding per example
        padded_moves: (B, n_max, d_move) — move features, zero-padded to n_max
        mask:         (B, n_max) bool — True where a real move sits
        returns:      (B, n_max) logits with padded slots set to -inf, so a
                      softmax / log_softmax over dim=1 excludes them. The head is
                      row-wise, so padding never contaminates real moves' logits.
        """
        B, n_max, d_move = padded_moves.shape
        d_hidden = pos_emb.shape[1]
        move_emb = self.move_encoder(
            padded_moves.reshape(B * n_max, d_move)
        ).reshape(B, n_max, d_hidden)
        pos_emb_b = pos_emb.unsqueeze(1).expand(B, n_max, d_hidden)
        combined = torch.cat([pos_emb_b, move_emb], dim=2)   # (B, n_max, 2*d_hidden)
        logits = self.head(
            combined.reshape(B * n_max, 2 * d_hidden)
        ).reshape(B, n_max)
        return logits.masked_fill(~mask, float("-inf"))

    def policy_and_value(
        self, position: torch.Tensor, moves: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One forward over the position trunk producing both heads' outputs."""
        if moves.dim() != 2:
            raise ValueError(f"moves must be (N, D); got {tuple(moves.shape)}")
        n = moves.shape[0]
        pos_emb = self._position_embedding(position)        # (1, d_hidden)
        wdl = torch.softmax(self.value_head(pos_emb), dim=-1)
        v = (wdl[..., 0] - wdl[..., 2]).reshape(())          # scalar Q for MCTS
        pos_emb_n = pos_emb.expand(n, -1)
        move_emb = self.move_encoder(moves)
        combined = torch.cat([pos_emb_n, move_emb], dim=1)
        logits = self.head(combined).squeeze(-1)            # (N,)
        return logits, v


class ChesskersScorerV2(nn.Module):
    """Square-grounded successor to ChesskersScorer (research-backed; see the
    `project-policy-head-redesign` memory).

    The defect in V1 is that it pools the board trunk to a single global vector
    BEFORE scoring any move, so a move can't see the board features at the
    squares it touches. V2 keeps the trunk output SPATIAL — a (c_filters, 10, 10)
    feature map F over the 8x8 board + 1-square rim — and computes each move's
    logit by GATHERING F at the squares the move touches:

      logit(m) = (W_src·F[from]) · (W_tgt·F[to]) / sqrt(d)      # Leela's source x
                                                                # target dot product
               + MLP([F[from], F[to], pathpool(F, waypoints), type_scalars])

    The dot-product term is exactly Leela Chess Zero's attention policy head
    (each logit depends only on its source + target square embeddings); the
    MLP term adds the chessckers-specific path context (a mean-pool of F over the
    intermediate rim-aware waypoint squares) and the non-spatial move-type
    scalars (capture/deploy/charge/promotion). Because chessckers' move generator
    ENUMERATES whole capture chains, V2 keeps V1's listwise "score each legal
    move" interface — forward / policy_and_value / batch_eval have identical
    signatures, so V2 is a drop-in swap for A/B against V1. Value + moves-left
    heads are unchanged (they legitimately pool — value is a global property).

    All move-logit paths funnel through `_gather_logits`, so MCTS priors and the
    training loss can't diverge (the property V1 advertised for _batched_move_logits).
    """

    def __init__(
        self,
        c_in: int = POS_C_V2,
        d_move: int = MOVE_D_V2,
        d_hidden: int = 256,
        c_filters: int = 96,
        n_blocks: int = 4,
    ):
        super().__init__()
        if d_move != MOVE_D_V2:
            raise ValueError(f"ChesskersScorerV2 expects d_move={MOVE_D_V2}; got {d_move}")
        self.c_filters = c_filters
        # Spatial trunk: initial conv + residual tower, NO flatten/pool. Output
        # stays (B, c_filters, 10, 10) so the policy head can gather per-square.
        self.position_trunk = nn.Sequential(
            nn.Conv2d(c_in, c_filters, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=c_filters),
            nn.ReLU(inplace=True),
            *[ResidualBlock(c_filters) for _ in range(n_blocks)],
        )
        # Value/moves-left path: global-pool the spatial map, then a bottleneck.
        self.value_trunk = nn.Sequential(
            nn.Linear(c_filters, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(inplace=True),
        )
        # Policy gather head: source/target projections for the dot product +
        # a context MLP over the three gathered feature vectors and the scalars.
        n_scalars = MOVE_D_V2 - MV2_SCALAR_BASE
        self.src_proj = nn.Linear(c_filters, d_hidden)
        self.tgt_proj = nn.Linear(c_filters, d_hidden)
        self.policy_scale = float(d_hidden) ** 0.5
        self.ctx_mlp = nn.Sequential(
            nn.Linear(3 * c_filters + n_scalars, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, 1),
        )
        # Same WDL value + softplus moves-left heads as V1.
        self.value_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden // 2, 3),
        )
        self.moves_left_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden // 2, 1),
            nn.Softplus(),
        )

    def _spatial(self, position: torch.Tensor) -> torch.Tensor:
        if position.dim() != 4 or position.shape[2:] != (10, 10):
            raise ValueError(f"position must be (B, C, 10, 10); got {tuple(position.shape)}")
        return self.position_trunk(position)  # (B, c_filters, 10, 10)

    def _pooled(self, spatial: torch.Tensor) -> torch.Tensor:
        return self.value_trunk(spatial.mean(dim=(2, 3)))  # (B, d_hidden)

    def _gather_logits(
        self, f_flat: torch.Tensor, padded_moves: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Per-move logits for a padded batch of moves, gathered from the spatial
        trunk. Shared by forward / policy_and_value / batch_eval / training so
        they can't diverge.

        f_flat:       (B, c_filters, 100) — the spatial map flattened over 10x10
        padded_moves: (B, n_max, MOVE_D_V2) — V2 move vectors, zero-padded
        mask:         (B, n_max) bool — True where a real move sits
        returns:      (B, n_max) logits, padded slots set to -inf.
        """
        B, n_max, _ = padded_moves.shape
        Cf = f_flat.shape[1]
        from_idx = padded_moves[:, :, 0].long()              # (B, n_max)
        to_idx = padded_moves[:, :, 1].long()
        path = padded_moves[:, :, MV2_PATH_BASE:MV2_SCALAR_BASE]  # (B, n_max, 100)
        scalars = padded_moves[:, :, MV2_SCALAR_BASE:]       # (B, n_max, K)

        # Gather F at the from/to squares (padded rows index 0 → masked out below).
        idx_from = from_idx.unsqueeze(1).expand(B, Cf, n_max)
        from_feat = torch.gather(f_flat, 2, idx_from).transpose(1, 2)  # (B, n_max, Cf)
        idx_to = to_idx.unsqueeze(1).expand(B, Cf, n_max)
        to_feat = torch.gather(f_flat, 2, idx_to).transpose(1, 2)
        # Mean-pool F over the intermediate waypoint squares.
        denom = path.sum(dim=-1, keepdim=True).clamp_min(1.0)
        path_feat = torch.bmm(path / denom, f_flat.transpose(1, 2))    # (B, n_max, Cf)

        src = self.src_proj(from_feat)
        tgt = self.tgt_proj(to_feat)
        dot = (src * tgt).sum(dim=-1) / self.policy_scale    # (B, n_max)
        ctx = self.ctx_mlp(
            torch.cat([from_feat, to_feat, path_feat, scalars], dim=-1)
        ).squeeze(-1)                                        # (B, n_max)
        return (dot + ctx).masked_fill(~mask, float("-inf"))

    def forward(self, position: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
        """position: (1, C, 10, 10); moves: (N, MOVE_D_V2) → (N,) policy logits."""
        if moves.dim() != 2:
            raise ValueError(f"moves must be (N, D); got {tuple(moves.shape)}")
        spatial = self._spatial(position)
        f_flat = spatial.reshape(1, self.c_filters, 100)
        n = moves.shape[0]
        valid = torch.ones((1, n), dtype=torch.bool, device=moves.device)
        return self._gather_logits(f_flat, moves.unsqueeze(0), valid).squeeze(0)

    def value(self, position: torch.Tensor) -> torch.Tensor:
        """Scalar Q = P(win) − P(loss) from the side-to-move's POV, in [-1, 1]."""
        wdl = torch.softmax(self.value_head(self._pooled(self._spatial(position))), dim=-1)
        return (wdl[..., 0] - wdl[..., 2]).reshape(())

    def policy_and_value(
        self, position: torch.Tensor, moves: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One trunk pass producing both heads — the MCTS leaf path."""
        if moves.dim() != 2:
            raise ValueError(f"moves must be (N, D); got {tuple(moves.shape)}")
        spatial = self._spatial(position)
        wdl = torch.softmax(self.value_head(self._pooled(spatial)), dim=-1)
        v = (wdl[..., 0] - wdl[..., 2]).reshape(())
        f_flat = spatial.reshape(1, self.c_filters, 100)
        n = moves.shape[0]
        valid = torch.ones((1, n), dtype=torch.bool, device=moves.device)
        logits = self._gather_logits(f_flat, moves.unsqueeze(0), valid).squeeze(0)
        return logits, v

    def batch_eval(
        self, positions: torch.Tensor, moves_list: list[torch.Tensor | None]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Batched eval mirroring V1.batch_eval: one trunk pass for B positions,
        per-position gather of the ragged move lists. Returns (values (B,),
        priors list of B (N_i,) softmaxed tensors)."""
        if positions.dim() != 4:
            raise ValueError(f"positions must be (B, C, 10, 10); got {tuple(positions.shape)}")
        if positions.shape[0] != len(moves_list):
            raise ValueError(
                f"batch size mismatch: positions B={positions.shape[0]}, "
                f"moves_list len={len(moves_list)}"
            )
        spatial = self._spatial(positions)                   # (B, Cf, 10, 10)
        B = positions.shape[0]
        f_flat = spatial.reshape(B, self.c_filters, 100)
        wdl = torch.softmax(self.value_head(self._pooled(spatial)), dim=-1)
        values = wdl[:, 0] - wdl[:, 2]                       # (B,)
        device = positions.device

        lengths = [0 if (m is None or m.shape[0] == 0) else int(m.shape[0]) for m in moves_list]
        n_max = max(lengths, default=0)
        if n_max == 0:
            return values, [torch.empty(0, device=device) for _ in range(B)]

        padded = positions.new_zeros((B, n_max, MOVE_D_V2))
        mask = torch.zeros((B, n_max), dtype=torch.bool, device=device)
        for i, moves in enumerate(moves_list):
            if lengths[i]:
                padded[i, : lengths[i]] = moves
                mask[i, : lengths[i]] = True

        logits = self._gather_logits(f_flat, padded, mask)
        priors = torch.softmax(logits, dim=1)                # (B, n_max); padding → 0
        return values, [
            priors[i, : lengths[i]] if lengths[i] else torch.empty(0, device=device)
            for i in range(B)
        ]
