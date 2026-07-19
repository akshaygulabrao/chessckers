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

from chessckers_engine.encoding import encode_move, encode_position, encoders_for
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
    policy_loss_fn: nn.Module | None = None,
    value_loss_fn: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-example reference loss: (policy CE, WDL value CE, moves-left MSE).
    Mirrors _batch_loss for a singleton batch (bit-identical). Reference/test
    only — the training loop uses _batch_loss. The *_loss_fn args are accepted
    for backward-compat but unused (WDL uses cross-entropy)."""
    device = next(model.parameters()).device
    pos = encode_position(example.fen).unsqueeze(0).to(device)
    moves = torch.stack([encode_move(m) for m in example.legal_moves]).to(device)
    target_dist = torch.tensor(example.visit_distribution, dtype=torch.float32, device=device)
    pos_emb = model.position_trunk(pos)                              # (1, d_hidden)
    n = moves.shape[0]
    logits = model.head(
        torch.cat([pos_emb.expand(n, -1), model.move_encoder(moves)], dim=1)
    ).squeeze(-1)
    policy_loss = -(target_dist * torch.log_softmax(logits, dim=0)).sum()
    wdl_logits = model.value_head(pos_emb)                           # (1, 3)
    wdl_target = torch.tensor([example.wdl_target], dtype=torch.float32, device=device)
    value_loss = -(wdl_target * torch.log_softmax(wdl_logits, dim=1)).sum(dim=1).mean()
    ml_pred = model.moves_left_head(pos_emb).squeeze(-1)
    ml_target = torch.tensor([example.moves_left_target], dtype=torch.float32, device=device)
    mlh_loss = nn.functional.mse_loss(ml_pred / MLH_TARGET_SCALE, ml_target / MLH_TARGET_SCALE)
    return policy_loss, value_loss, mlh_loss


# Normalize the moves-left target (~game length in plies) so its MSE sits at
# ~O(1), comparable to the policy/value losses.
MLH_TARGET_SCALE = 40.0


def _discounted_wdl(wdl: list[float], moves_left: float, gamma: float) -> list[float]:
    """Pull a WDL outcome target toward the DRAW vertex by gamma**(plies_to_end-1),
    so a position N plies from the win keeps only gamma**(N-1) of its decisive mass
    and the rest moves to draw. This makes a FASTER win a strictly stronger target
    than a slow one, giving the value head (and hence MCTS + policy) an incentive to
    convert quickly — which flat ±1 targets cannot express. gamma>=1 => unchanged.

    Generic over any soft WDL target: new = g·wdl + (1-g)·[0,1,0], which always stays
    a valid distribution (sums to 1). The position one ply from the end (moves_left=1)
    keeps full strength (g=1)."""
    if gamma >= 1.0:
        return wdl
    g = gamma ** max(moves_left - 1.0, 0.0)
    w, d, l = wdl
    return [g * w, g * d + (1.0 - g), g * l]


def _value_target(
    wdl: list[float],
    search_wdl: list[float] | None,
    moves_left: float,
    gamma: float,
    q_ratio: float,
) -> list[float]:
    """Blend the (gamma-discounted) OUTCOME target z with the SEARCH's root value q:
    ``(1 - q_ratio)·z + q_ratio·search_wdl``. The discount shapes z only (an incentive
    to win faster); q is the search's position estimate, used raw. q_ratio>0 pulls the
    value target toward what the search expects under best play, which removes the
    conservatism that temperature / Dirichlet noise bake into the realized outcome.
    Falls back to pure z when q_ratio<=0 or this example carries no search value
    (search_wdl is None), so old / scalar-only chunks are unaffected."""
    z = _discounted_wdl(wdl, moves_left, gamma)
    if q_ratio <= 0.0 or search_wdl is None:
        return z
    return [(1.0 - q_ratio) * z[k] + q_ratio * search_wdl[k] for k in range(3)]


def _batch_loss(
    model: ChesskersScorer,
    batch: list[AZExample],
    value_loss_fn: nn.Module | None = None,
    gamma: float = 1.0,
    q_ratio: float = 0.0,
    improved_policy: bool = False,
    diag: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mini-batched loss: trunk batched in one forward; WDL value head (cross-
    entropy vs the one-hot outcome) + moves-left head (MSE on scaled plies-to-
    end) batched; policy head per-position (ragged move counts). Returns
    (policy_loss, value_loss, mlh_loss), each a mean over the batch.
    `value_loss_fn` is accepted for backward-compat but unused (WDL uses CE).

    If `diag` is a dict, it is filled (under no_grad, from the SAME outputs/targets
    — the returned losses are bit-identical whether or not diag is passed) with two
    cheap anchored progress signals, robust under value-target shift:
      value_sign_agree  — fraction where sign(pred win−loss) == sign(target win−loss)
      policy_top1_agree — fraction where argmax(policy logits) == argmax(policy target)
    The continuous trainer passes diag only on log-due steps, so this is ~free."""
    device = next(model.parameters()).device
    # Encoders follow the model's arch VERSION (V1 8×8 / 240-d moves, V2 10×10 /
    # 114-d gather-indexed moves); everything else is version-uniform via
    # training_forward, so this loss is bit-identical to the old V1 inline body.
    enc_pos, _, enc_move = encoders_for(getattr(model, "VERSION", "v1"))
    positions = torch.stack(
        [enc_pos(ex.fen) for ex in batch]
    ).to(device)                                                # (B, C, H, W)

    # Build the padded moves + mask + visit-distribution target on CPU, then ONE
    # host->device transfer each (mirrors how `positions` is built above).
    move_lists = [
        torch.stack([enc_move(m) for m in ex.legal_moves]) for ex in batch
    ]                                                            # B × (N_i, D) on CPU
    counts = [mv.shape[0] for mv in move_lists]
    B, n_max, d_move = len(batch), max(counts), move_lists[0].shape[1]
    padded = torch.zeros((B, n_max, d_move), dtype=torch.float32)
    mask = torch.zeros((B, n_max), dtype=torch.bool)
    target = torch.zeros((B, n_max), dtype=torch.float32)
    for i, (mv, ex) in enumerate(zip(move_lists, batch)):
        padded[i, : counts[i]] = mv
        mask[i, : counts[i]] = True
        # Gumbel: train the policy on the search's improved policy when requested and
        # present; fall back to the visit distribution for pre-Gumbel examples.
        pi = (ex.improved_policy if improved_policy and ex.improved_policy is not None
              else ex.visit_distribution)
        target[i, : counts[i]] = torch.tensor(pi, dtype=torch.float32)
    padded, mask, target = padded.to(device), mask.to(device), target.to(device)

    # One trunk pass → policy logits + WDL value logits + moves-left preds. The
    # policy head is the SAME padded forward MCTS uses, so training loss and
    # inference priors can't drift apart.
    logits, wdl_logits, ml_pred = model.training_forward(positions, padded, mask)

    # WDL value: cross-entropy against the (optionally z↔q blended) Win/Draw/Loss target.
    wdl_targets = torch.tensor(
        [_value_target(ex.wdl_target, ex.search_wdl, ex.moves_left_target, gamma, q_ratio)
         for ex in batch],
        dtype=torch.float32, device=device,
    )                                                            # (B, 3), soft when gamma<1 or q_ratio>0
    value_loss = -(wdl_targets * torch.log_softmax(wdl_logits, dim=1)).sum(dim=1).mean()
    # Moves-left: MSE on scaled plies-to-end.
    ml_target = torch.tensor(
        [ex.moves_left_target for ex in batch], dtype=torch.float32, device=device,
    )                                                            # (B,)
    mlh_loss = nn.functional.mse_loss(ml_pred / MLH_TARGET_SCALE, ml_target / MLH_TARGET_SCALE)
    # Policy CE per example = -Σ target·log_probs over valid moves; padded slots
    # have target=0 and log_probs=-inf (0·-inf=NaN), so zero them before summing.
    log_probs = torch.log_softmax(logits, dim=1)
    policy_loss = (-(target * log_probs)).masked_fill(~mask, 0.0).sum(dim=1).mean()

    if diag is not None:
        with torch.no_grad():
            # Value sign: scalar expected value = P(win) − P(loss) for both the
            # predicted WDL and the (possibly soft/blended) target. Comparing only
            # the SIGN is stable under target-shift — magnitude/calibration can
            # drift while "who's winning" stays anchored.
            pred_wdl = torch.softmax(wdl_logits, dim=1)
            pred_v = pred_wdl[:, 0] - pred_wdl[:, 2]
            tgt_v = wdl_targets[:, 0] - wdl_targets[:, 2]
            diag["value_sign_agree"] = float(
                (torch.sign(pred_v) == torch.sign(tgt_v)).float().mean())
            # Policy top-1: argmax over VALID moves only (padded slots forced to
            # -inf for the logits; target is already 0 on padding so its argmax
            # lands on a real move).
            masked_logits = logits.masked_fill(~mask, float("-inf"))
            diag["policy_top1_agree"] = float(
                (masked_logits.argmax(dim=1) == target.argmax(dim=1)).float().mean())

    return policy_loss, value_loss, mlh_loss


def train_az(
    model: ChesskersScorer,
    examples: list[AZExample],
    epochs: int = 5,
    lr: float = 1e-3,
    seed: int = 0,
    log_every: int = 0,
    value_loss_weight: float = 1.0,
    mlh_loss_weight: float = 0.3,
    grad_clip: float | None = 1.0,
    batch_size: int = 1,
) -> AZTrainResult:
    """Run dual-loss training over `examples` for `epochs` passes.

    `batch_size`=1 reproduces the per-example loop (one optimizer step per
    example). Larger values use mini-batched forward+backward, which is
    much faster on GPU because the trunk + value head batch naturally."""
    torch.manual_seed(seed)
    if not examples:
        return AZTrainResult(epoch_losses=[{"policy": 0.0, "value": 0.0, "mlh": 0.0, "total": 0.0}] * epochs,
                             final_loss=0.0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
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
        running_m = 0.0
        for step, batch_start in enumerate(range(0, len(examples), bs)):
            batch_idx = idxs[batch_start:batch_start + bs]
            batch = [examples[i] for i in batch_idx]
            opt.zero_grad()
            p_loss, v_loss, m_loss = _batch_loss(model, batch)
            total = p_loss + value_loss_weight * v_loss + mlh_loss_weight * m_loss
            total.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
            running_p += float(p_loss.item()) * len(batch)
            running_v += float(v_loss.item()) * len(batch)
            running_m += float(m_loss.item()) * len(batch)
            if log_every and (step + 1) % log_every == 0:
                log.info("epoch %d step %d policy=%.4f value=%.4f mlh=%.4f",
                         ep + 1, step + 1, p_loss.item(), v_loss.item(), m_loss.item())
        n = len(examples)
        avg_p = running_p / n
        avg_v = running_v / n
        avg_m = running_m / n
        total_avg = avg_p + value_loss_weight * avg_v + mlh_loss_weight * avg_m
        epoch_losses.append({"policy": avg_p, "value": avg_v, "mlh": avg_m, "total": total_avg})
        log.info("epoch %d done: policy=%.4f value=%.4f mlh=%.4f total=%.4f",
                 ep + 1, avg_p, avg_v, avg_m, total_avg)

    return AZTrainResult(epoch_losses=epoch_losses, final_loss=epoch_losses[-1]["total"])


def save_checkpoint(model: ChesskersScorer, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    # Self-describing sidecar: the bare `.pt` stays a plain state_dict (every
    # existing load_checkpoint reader is untouched), but an `<path>.arch.json`
    # records the trunk recipe so checkpoints.load_scorer can rebuild the exact
    # architecture — essential once the trunk is non-default (e.g. the V2
    # transformer). Written only for build_model-constructed models (which carry
    # `.arch`); models built directly get no sidecar and load as before.
    arch = getattr(model, "arch", None)
    if arch is not None:
        import json
        Path(str(path) + ".arch.json").write_text(json.dumps(arch))
