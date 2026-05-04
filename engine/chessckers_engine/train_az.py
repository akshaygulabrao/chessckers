"""Dual-loss training for AlphaZero self-play examples.

Each `AZExample` has:
- a position (fen)
- a list of legal moves
- a target visit distribution over those moves (policy target)
- a scalar value target ∈ {-1, 0, 1} (value target)

Loss = policy_cross_entropy + value_mse, both computed per-example because
different positions have different legal-move counts (no clean batching of
ragged outputs without padding).

The training loop reuses the model's `policy_and_value()` so the position
trunk is run once per example and shared between heads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from chessckers_engine.encoding import encode_move, encode_position
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.selfplay_az import AZExample

log = logging.getLogger("chessckers_engine.train_az")


@dataclass
class AZTrainResult:
    epoch_losses: list[dict]   # per epoch: {"policy": ..., "value": ..., "total": ...}
    final_loss: float


def _example_loss(
    model: ChesskersScorer,
    example: AZExample,
    policy_loss_fn: nn.Module,
    value_loss_fn: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    pos = encode_position(example.fen).unsqueeze(0).to(device)
    moves = torch.stack([encode_move(m) for m in example.legal_moves]).to(device)
    target_dist = torch.tensor(example.visit_distribution, dtype=torch.float32, device=device)
    target_value = torch.tensor(example.value_target, dtype=torch.float32, device=device)

    logits, value = model.policy_and_value(pos, moves)
    log_probs = torch.log_softmax(logits, dim=0)
    # Cross-entropy with a soft target: -sum(target_i * log_prob_i)
    policy_loss = -(target_dist * log_probs).sum()
    value_loss = value_loss_fn(value, target_value)
    return policy_loss, value_loss


def _batch_loss(
    model: ChesskersScorer,
    batch: list[AZExample],
    value_loss_fn: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mini-batched loss: trunk + value head batched in one forward; policy
    head runs per-position because legal-move counts are ragged. Returns
    (policy_loss, value_loss) — both means over the batch (so the gradient
    magnitude is independent of batch size, matching standard mini-batch SGD)."""
    device = next(model.parameters()).device
    positions = torch.stack(
        [encode_position(ex.fen) for ex in batch]
    ).to(device)                                                # (B, C, 8, 8)
    target_values = torch.tensor(
        [ex.value_target for ex in batch],
        dtype=torch.float32, device=device,
    )                                                            # (B,)

    # Trunk + value head, batched.
    pos_emb = model.position_trunk(positions)                    # (B, d_hidden)
    values = model.value_head(pos_emb).squeeze(-1)               # (B,)
    value_loss = value_loss_fn(values, target_values)            # mean MSE

    # Policy head per-position (ragged moves). Sum the per-example
    # cross-entropies, then average — matches the per-example loss when B=1.
    policy_loss = torch.zeros((), device=device)
    for i, ex in enumerate(batch):
        moves = torch.stack(
            [encode_move(m) for m in ex.legal_moves]
        ).to(device)                                             # (N_i, D)
        target_dist = torch.tensor(
            ex.visit_distribution, dtype=torch.float32, device=device,
        )                                                        # (N_i,)
        n = moves.shape[0]
        pe = pos_emb[i:i + 1].expand(n, -1)
        me = model.move_encoder(moves)
        combined = torch.cat([pe, me], dim=1)
        logits = model.head(combined).squeeze(-1)                # (N_i,)
        log_probs = torch.log_softmax(logits, dim=0)
        policy_loss = policy_loss + -(target_dist * log_probs).sum()
    policy_loss = policy_loss / len(batch)

    return policy_loss, value_loss


def train_az(
    model: ChesskersScorer,
    examples: list[AZExample],
    epochs: int = 5,
    lr: float = 1e-3,
    seed: int = 0,
    log_every: int = 0,
    value_loss_weight: float = 1.0,
    grad_clip: float | None = 1.0,
    batch_size: int = 1,
) -> AZTrainResult:
    """Run dual-loss training over `examples` for `epochs` passes.

    `batch_size`=1 reproduces the per-example loop (one optimizer step per
    example). Larger values use mini-batched forward+backward, which is
    much faster on GPU because the trunk + value head batch naturally."""
    torch.manual_seed(seed)
    if not examples:
        return AZTrainResult(epoch_losses=[{"policy": 0.0, "value": 0.0, "total": 0.0}] * epochs,
                             final_loss=0.0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    value_loss_fn = nn.MSELoss()
    epoch_losses: list[dict] = []

    model.train()
    rng = torch.Generator().manual_seed(seed)
    bs = max(1, batch_size)

    for ep in range(epochs):
        idxs = torch.randperm(len(examples), generator=rng).tolist()
        # Track per-example loss totals so the avg is comparable across
        # different batch sizes. _batch_loss returns the *mean* per-example
        # loss for the batch, so multiply back by len(batch) to recover the sum.
        running_p = 0.0
        running_v = 0.0
        for step, batch_start in enumerate(range(0, len(examples), bs)):
            batch_idx = idxs[batch_start:batch_start + bs]
            batch = [examples[i] for i in batch_idx]
            opt.zero_grad()
            p_loss, v_loss = _batch_loss(model, batch, value_loss_fn)
            total = p_loss + value_loss_weight * v_loss
            total.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
            running_p += float(p_loss.item()) * len(batch)
            running_v += float(v_loss.item()) * len(batch)
            if log_every and (step + 1) % log_every == 0:
                log.info("epoch %d step %d policy=%.4f value=%.4f",
                         ep + 1, step + 1, p_loss.item(), v_loss.item())
        n = len(examples)
        avg_p = running_p / n
        avg_v = running_v / n
        epoch_losses.append({"policy": avg_p, "value": avg_v, "total": avg_p + value_loss_weight * avg_v})
        log.info("epoch %d done: policy=%.4f value=%.4f total=%.4f",
                 ep + 1, avg_p, avg_v, epoch_losses[-1]["total"])

    return AZTrainResult(epoch_losses=epoch_losses, final_loss=epoch_losses[-1]["total"])


def save_checkpoint(model: ChesskersScorer, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
