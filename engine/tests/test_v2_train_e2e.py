"""End-to-end smoke test for the V2 (square-grounded) arch through the real
self-play → training path: play_az_game drives PUCT MCTS with a ChesskersScorerV2
(so mcts_puct._evaluate_leaf picks the V2 10x10 / gather-indexed encoders by the
model's VERSION tag), the game becomes AZExamples, and train_az runs a step
(so _batch_loss → model.training_forward → _gather_logits). Proves V2 trains
through the same machinery V1 uses — the precondition for a V1-vs-V2 A/B run.

Marked slow: it plays a (short) MCTS self-play game.
"""

import pytest
import torch

from chessckers_engine.model import build_model
from chessckers_engine.selfplay_az import az_game_to_examples, play_az_game
from chessckers_engine.train_az import train_az
from chessckers_engine.variant_py import PyVariantClient

pytestmark = pytest.mark.slow


def test_v2_selfplay_then_train_runs_and_steps():
    torch.manual_seed(0)
    model = build_model(version="v2", d_hidden=32, c_filters=16, n_blocks=1)
    assert model.VERSION == "v2"
    client = PyVariantClient()

    # Short self-play game: low sims, capped plies — we want signal that the
    # path runs, not a strong game.
    rng = torch.Generator().manual_seed(0)
    game = play_az_game(
        model, client, n_sims=12, c_puct=1.5, temperature=1.0,
        max_plies=16, rng=rng, dirichlet_alpha=0.3,
    )
    examples = az_game_to_examples(game)
    assert examples, "self-play produced no training examples"

    # A real training step must run and actually move the weights.
    before = [p.detach().clone() for p in model.parameters()]
    result = train_az(model, examples, epochs=1, lr=1e-2, batch_size=8)
    assert result.epoch_losses
    final = result.epoch_losses[-1]
    for k in ("policy", "value", "mlh", "total"):
        assert torch.isfinite(torch.tensor(final[k])), f"{k} loss not finite: {final}"
    moved = any(
        not torch.equal(b, p) for b, p in zip(before, model.parameters())
    )
    assert moved, "no parameter changed — training step did not run"

    client.close()
