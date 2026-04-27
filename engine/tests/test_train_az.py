import torch

from chessckers_engine.model import ChesskersScorer
from chessckers_engine.selfplay_az import AZExample
from chessckers_engine.train_az import save_checkpoint, train_az

INITIAL_FEN = (
    "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"
    "[a6:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,"
    "a7:k,b7:k,c7:k,d7:k,e7:k,f7:k,g7:k,h7:k,"
    "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] w KQkq - 0 1"
)


_FILES = "abcdefgh"


def _move_distinct(idx: int) -> dict:
    """Each idx gets a unique (from, to) one-hot in the move encoding so the
    model can actually learn to discriminate between candidates."""
    f = _FILES[idx % 8]
    t = _FILES[(idx + 1) % 8]
    return {"uci": f"M{idx}", "from": f"{f}2", "to": f"{t}3"}


def _example(visit_dist: list[float], value: float) -> AZExample:
    moves = [_move_distinct(i) for i in range(len(visit_dist))]
    return AZExample(
        fen=INITIAL_FEN,
        legal_moves=moves,
        visit_distribution=visit_dist,
        value_target=value,
    )


def test_total_loss_decreases_on_constant_targets():
    """Repeated training on the same target should monotonically reduce total loss."""
    torch.manual_seed(0)
    model = ChesskersScorer()
    examples = [_example([0.1, 0.7, 0.2], 0.5) for _ in range(32)]
    result = train_az(model, examples, epochs=8, lr=5e-3, log_every=0)
    # Just confirm the trend is downward; the magnitude is dominated by entropy
    # of the soft-target cross-entropy and bounded below by the target's entropy.
    assert result.epoch_losses[-1]["total"] < result.epoch_losses[0]["total"]


def test_policy_loss_responds_to_one_hot_target():
    """With a sharp one-hot policy target on a single example, the policy loss
    should converge to near zero given enough epochs."""
    torch.manual_seed(0)
    model = ChesskersScorer()
    examples = [_example([0.0, 1.0, 0.0], 0.0)] * 32  # policy target sharply favors move 1
    result = train_az(model, examples, epochs=20, lr=5e-3, log_every=0)
    # The cross-entropy loss for a perfect prediction is 0; we should be much smaller than the start.
    assert result.epoch_losses[-1]["policy"] < 0.5


def test_value_loss_responds_to_constant_value_target():
    torch.manual_seed(0)
    model = ChesskersScorer()
    examples = [_example([1.0], 0.8) for _ in range(16)]
    result = train_az(model, examples, epochs=20, lr=5e-3, log_every=0)
    # Tanh's output saturates near ±1, so reaching 0.8 is feasible.
    assert result.epoch_losses[-1]["value"] < 0.1


def test_empty_examples_returns_zero_losses_without_crashing():
    model = ChesskersScorer()
    result = train_az(model, [], epochs=3, log_every=0)
    assert all(e["total"] == 0.0 for e in result.epoch_losses)


def test_checkpoint_save_and_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = ChesskersScorer()
    examples = [_example([0.5, 0.5], 0.0) for _ in range(8)]
    train_az(model, examples, epochs=3, log_every=0)
    path = tmp_path / "az.pt"
    save_checkpoint(model, path)

    fresh = ChesskersScorer()
    fresh.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    pos = torch.zeros(1, 14, 8, 8)
    mv = torch.zeros(1, 140)
    with torch.no_grad():
        a_logits, a_value = model.policy_and_value(pos, mv)
        b_logits, b_value = fresh.policy_and_value(pos, mv)
    assert torch.allclose(a_logits, b_logits)
    assert torch.allclose(a_value, b_value)
