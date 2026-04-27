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
    pos = encode_position(example.fen).unsqueeze(0)
    moves = torch.stack([encode_move(m) for m in example.legal_moves])
    target_dist = torch.tensor(example.visit_distribution, dtype=torch.float32)
    target_value = torch.tensor(example.value_target, dtype=torch.float32)

    logits, value = model.policy_and_value(pos, moves)
    log_probs = torch.log_softmax(logits, dim=0)
    # Cross-entropy with a soft target: -sum(target_i * log_prob_i)
    policy_loss = -(target_dist * log_probs).sum()
    value_loss = value_loss_fn(value, target_value)
    return policy_loss, value_loss


def train_az(
    model: ChesskersScorer,
    examples: list[AZExample],
    epochs: int = 5,
    lr: float = 1e-3,
    seed: int = 0,
    log_every: int = 0,
    value_loss_weight: float = 1.0,
) -> AZTrainResult:
    """Run dual-loss training over `examples` for `epochs` passes."""
    torch.manual_seed(seed)
    if not examples:
        return AZTrainResult(epoch_losses=[{"policy": 0.0, "value": 0.0, "total": 0.0}] * epochs,
                             final_loss=0.0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    value_loss_fn = nn.MSELoss()
    epoch_losses: list[dict] = []

    model.train()
    rng = torch.Generator().manual_seed(seed)

    for ep in range(epochs):
        idxs = torch.randperm(len(examples), generator=rng).tolist()
        running_p = 0.0
        running_v = 0.0
        for step, i in enumerate(idxs):
            ex = examples[i]
            opt.zero_grad()
            p_loss, v_loss = _example_loss(model, ex, None, value_loss_fn)
            total = p_loss + value_loss_weight * v_loss
            total.backward()
            opt.step()
            running_p += float(p_loss.item())
            running_v += float(v_loss.item())
            if log_every and (step + 1) % log_every == 0:
                log.info("epoch %d step %d policy=%.4f value=%.4f", ep + 1, step + 1, p_loss.item(), v_loss.item())
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
