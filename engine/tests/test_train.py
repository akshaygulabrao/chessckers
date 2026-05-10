import torch

from chessckers_engine.encoding import MOVE_D
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.train import save_checkpoint, train

INITIAL_FEN = (
    "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"
    "[a6:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,"
    "a7:k,b7:k,c7:k,d7:k,e7:k,f7:k,g7:k,h7:k,"
    "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] w KQkq - 0 1"
)


def _synthetic_examples(n: int, target_value: float) -> list[dict]:
    return [
        {"fen": INITIAL_FEN, "move": {"from": "e2", "to": "e4", "uci": "e2e4"}, "target": target_value}
        for _ in range(n)
    ]


def test_loss_decreases_on_a_constant_target():
    """Trivial overfit: every example has target=5.0. After a few epochs, loss
    should be much smaller than at epoch 1 (the network learns to output 5.0)."""
    torch.manual_seed(0)
    model = ChesskersScorer()
    examples = _synthetic_examples(64, target_value=5.0)
    result = train(model, examples, epochs=8, batch_size=16, log_every=0)
    assert result.epoch_losses[-1] < result.epoch_losses[0] * 0.5


def test_can_overfit_a_handful_of_examples_to_low_loss():
    torch.manual_seed(0)
    model = ChesskersScorer()
    examples = _synthetic_examples(8, target_value=2.0)
    result = train(model, examples, epochs=30, batch_size=8, lr=5e-3, log_every=0)
    # On 8 identical examples, the network should converge close to target=2.0
    assert result.final_loss < 0.5


def test_checkpoint_save_and_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = ChesskersScorer()
    examples = _synthetic_examples(8, target_value=3.0)
    train(model, examples, epochs=5, batch_size=8, log_every=0)

    path = tmp_path / "weights.pt"
    save_checkpoint(model, path)
    assert path.exists()

    fresh = ChesskersScorer()
    fresh.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    # Same forward pass output before and after roundtrip.
    pos = torch.zeros(1, 14, 8, 8)
    mv = torch.zeros(1, MOVE_D)
    with torch.no_grad():
        a = model(pos, mv)
        b = fresh(pos, mv)
    assert torch.allclose(a, b)


def test_empty_dataset_does_not_crash():
    model = ChesskersScorer()
    result = train(model, [], epochs=3, batch_size=8, log_every=0)
    # With no examples there are no batches; epoch losses stay 0 by convention.
    assert result.epoch_losses == [0.0, 0.0, 0.0]
