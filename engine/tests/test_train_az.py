import torch

from chessckers_engine.encoding import MOVE_D
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.selfplay_az import AZExample
from chessckers_engine.train_az import _batch_loss, _example_loss, save_checkpoint, train_az

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


def _example(visit_dist: list[float], wdl=(0.0, 1.0, 0.0), moves_left: float = 10.0) -> AZExample:
    """WDL value target (default = draw one-hot); moves_left = plies-to-end."""
    moves = [_move_distinct(i) for i in range(len(visit_dist))]
    return AZExample(
        fen=INITIAL_FEN,
        legal_moves=moves,
        visit_distribution=visit_dist,
        wdl_target=list(wdl),
        moves_left_target=moves_left,
    )


def test_total_loss_decreases_on_constant_targets():
    """Repeated training on the same target should monotonically reduce total loss."""
    torch.manual_seed(0)
    model = ChesskersScorer()
    examples = [_example([0.1, 0.7, 0.2]) for _ in range(32)]
    result = train_az(model, examples, epochs=8, lr=5e-3, log_every=0)
    assert result.epoch_losses[-1]["total"] < result.epoch_losses[0]["total"]


def test_policy_loss_responds_to_one_hot_target():
    """With a sharp one-hot policy target on a single example, the policy loss
    should converge to near zero given enough epochs."""
    torch.manual_seed(0)
    model = ChesskersScorer()
    examples = [_example([0.0, 1.0, 0.0])] * 32  # policy target sharply favors move 1
    result = train_az(model, examples, epochs=20, lr=5e-3, log_every=0)
    assert result.epoch_losses[-1]["policy"] < 0.5


def test_value_loss_responds_to_constant_wdl_target():
    """WDL value head: a constant one-hot outcome target should drive the
    cross-entropy value loss toward 0 (a one-hot is exactly reachable)."""
    torch.manual_seed(0)
    model = ChesskersScorer()
    examples = [_example([1.0], wdl=(1.0, 0.0, 0.0)) for _ in range(16)]  # always "win"
    result = train_az(model, examples, epochs=20, lr=5e-3, log_every=0)
    assert result.epoch_losses[-1]["value"] < 0.1


def test_empty_examples_returns_zero_losses_without_crashing():
    model = ChesskersScorer()
    result = train_az(model, [], epochs=3, log_every=0)
    assert all(e["total"] == 0.0 for e in result.epoch_losses)


def test_grad_clip_keeps_weights_finite_under_aggressive_lr():
    """At a learning rate high enough to cause numerical blow-up, clipping
    should keep the model's parameters finite (no NaN / Inf)."""
    torch.manual_seed(0)
    examples = [_example([0.0, 1.0, 0.0]) for _ in range(16)]

    torch.manual_seed(0)
    clipped = ChesskersScorer()
    train_az(clipped, examples, epochs=4, lr=1.0, log_every=0, grad_clip=1.0)
    for p in clipped.parameters():
        assert torch.isfinite(p.data).all().item(), "clipping should prevent NaN/Inf"


def test_grad_clip_disabled_with_none_or_zero():
    """Sanity: a model trained with grad_clip=None and one with grad_clip=0
    behave identically (both skip clipping)."""
    torch.manual_seed(0)
    examples = [_example([0.4, 0.6]) for _ in range(8)]

    torch.manual_seed(0)
    a = ChesskersScorer()
    train_az(a, examples, epochs=2, lr=1e-3, log_every=0, grad_clip=None)

    torch.manual_seed(0)
    b = ChesskersScorer()
    train_az(b, examples, epochs=2, lr=1e-3, log_every=0, grad_clip=0)

    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa.data, pb.data)


def test_checkpoint_save_and_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = ChesskersScorer()
    examples = [_example([0.5, 0.5]) for _ in range(8)]
    train_az(model, examples, epochs=3, log_every=0)
    path = tmp_path / "az.pt"
    save_checkpoint(model, path)

    fresh = ChesskersScorer()
    fresh.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    pos = torch.zeros(1, 14, 8, 8)
    mv = torch.zeros(1, MOVE_D)
    with torch.no_grad():
        a_logits, a_value = model.policy_and_value(pos, mv)
        b_logits, b_value = fresh.policy_and_value(pos, mv)
    assert torch.allclose(a_logits, b_logits)
    assert torch.allclose(a_value, b_value)


# ---- Mini-batched loss correctness (policy CE, WDL value CE, moves-left MSE) ----


def test_batch_loss_size_one_matches_example_loss():
    """_batch_loss with a singleton batch must match _example_loss exactly
    across all three losses — same math, just expressed batch-aware."""
    torch.manual_seed(0)
    model = ChesskersScorer().eval()
    ex = _example([0.1, 0.7, 0.2], wdl=(0.2, 0.3, 0.5), moves_left=12.0)
    p1, v1, m1 = _example_loss(model, ex)
    p2, v2, m2 = _batch_loss(model, [ex])
    assert abs(p1.item() - p2.item()) < 1e-5
    assert abs(v1.item() - v2.item()) < 1e-5
    assert abs(m1.item() - m2.item()) < 1e-5


def test_batch_loss_mean_equals_per_example_average():
    """_batch_loss returns the mean per-example loss; verify it equals the
    average of independently computed _example_loss values (all three heads)."""
    torch.manual_seed(0)
    model = ChesskersScorer().eval()
    examples = [
        _example([0.5, 0.3, 0.2], wdl=(1.0, 0.0, 0.0), moves_left=8.0),
        _example([0.1, 0.8, 0.1], wdl=(0.0, 0.0, 1.0), moves_left=20.0),
        _example([0.25, 0.25, 0.5], wdl=(0.0, 1.0, 0.0), moves_left=14.0),
    ]
    p_b, v_b, m_b = _batch_loss(model, examples)
    indiv = [_example_loss(model, ex) for ex in examples]
    for idx, agg in enumerate((p_b, v_b, m_b)):
        avg = sum(t[idx].item() for t in indiv) / len(indiv)
        assert abs(agg.item() - avg) < 1e-5


def test_train_az_with_batch_size_decreases_loss():
    """End-to-end: training with batch_size > 1 should still converge."""
    torch.manual_seed(0)
    model = ChesskersScorer()
    examples = [_example([0.1, 0.7, 0.2]) for _ in range(64)]
    result = train_az(model, examples, epochs=4, lr=5e-3, batch_size=16, log_every=0)
    assert result.epoch_losses[-1]["total"] < result.epoch_losses[0]["total"]
