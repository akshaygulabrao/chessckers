"""Supervised training of ChesskersScorer against material targets.

Loads (fen, move, target) examples from JSONL, encodes them lazily, trains
the model with MSE on the per-move logits, and saves a checkpoint that can
be loaded back via ENGINE_MODEL=path/to/weights.pt.

Run as a module:
    uv run python -m chessckers_engine.train \
        --data path/to/dataset.jsonl \
        --out  path/to/weights.pt \
        --epochs 10 \
        --batch-size 64
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from chessckers_engine.checkpoints import default_checkpoint_path
from chessckers_engine.dataset import Example, load_jsonl
from chessckers_engine.encoding import encode_move, encode_position
from chessckers_engine.model import ChesskersScorer

log = logging.getLogger("chessckers_engine.train")


class ExampleDataset(Dataset):
    """Lazy encoder over a list of (fen, move, target) examples."""

    def __init__(self, examples: list[Example]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            "position": encode_position(ex["fen"]),  # (C, 8, 8)
            "move": encode_move(ex["move"]),         # (D,)
            "target": torch.tensor(ex["target"], dtype=torch.float32),
        }


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "positions": torch.stack([b["position"] for b in batch]),  # (B, C, 8, 8)
        "moves": torch.stack([b["move"] for b in batch]),          # (B, D)
        "targets": torch.stack([b["target"] for b in batch]),      # (B,)
    }


def _forward_batch(model: ChesskersScorer, positions: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
    """Score one move per position. The scorer expects a single position +
    multiple candidate moves, but here each example has its own position. We
    run them as a loop over the batch (B small enough this is fine on CPU).
    Returns a (B,) tensor of logits."""
    logits = []
    for i in range(positions.shape[0]):
        pos = positions[i].unsqueeze(0)              # (1, C, 8, 8)
        mv = moves[i].unsqueeze(0)                   # (1, D)
        logits.append(model(pos, mv).squeeze(0))     # ()
    return torch.stack(logits)                       # (B,)


@dataclass
class TrainResult:
    epoch_losses: list[float]
    final_loss: float


def train(
    model: ChesskersScorer,
    examples: list[Example],
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    log_every: int = 50,
) -> TrainResult:
    """Run MSE training for `epochs` over `examples`. Returns per-epoch losses."""
    torch.manual_seed(seed)
    if not examples:
        log.warning("train() called with no examples; returning zero losses")
        return TrainResult(epoch_losses=[0.0] * epochs, final_loss=0.0)
    loader = DataLoader(ExampleDataset(examples), batch_size=batch_size, shuffle=True, collate_fn=_collate)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    epoch_losses: list[float] = []
    model.train()
    step = 0
    for epoch in range(epochs):
        running = 0.0
        n_batches = 0
        for batch in loader:
            opt.zero_grad()
            logits = _forward_batch(model, batch["positions"], batch["moves"])
            loss = loss_fn(logits, batch["targets"])
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
            step += 1
            if log_every and step % log_every == 0:
                log.info("epoch %d step %d loss=%.4f", epoch + 1, step, loss.item())
        epoch_loss = running / max(n_batches, 1)
        epoch_losses.append(epoch_loss)
        log.info("epoch %d done: avg_loss=%.4f", epoch + 1, epoch_loss)

    return TrainResult(epoch_losses=epoch_losses, final_loss=epoch_losses[-1] if epoch_losses else 0.0)


def save_checkpoint(model: ChesskersScorer, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="JSONL dataset path")
    p.add_argument(
        "--out",
        default=None,
        help="output .pt checkpoint path (default: engine/weights/model-<timestamp>.pt)",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_path = Path(args.out) if args.out else default_checkpoint_path()

    examples = load_jsonl(args.data)
    log.info("loaded %d examples from %s", len(examples), args.data)

    model = ChesskersScorer()
    result = train(model, examples, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    save_checkpoint(model, out_path)
    log.info("saved checkpoint to %s; final epoch loss=%.4f", out_path, result.final_loss)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
