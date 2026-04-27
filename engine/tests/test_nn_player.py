import torch

from chessckers_engine.model import ChesskersScorer
from chessckers_engine.nn_player import pick_nn

INITIAL_FEN = (
    "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"
    "[a6:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,"
    "a7:k,b7:k,c7:k,d7:k,e7:k,f7:k,g7:k,h7:k,"
    "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] w KQkq - 0 1"
)


def _state(moves: list[dict]) -> dict:
    return {"fen": INITIAL_FEN, "legalMoves": moves}


def test_returns_one_of_the_legal_moves():
    torch.manual_seed(0)
    model = ChesskersScorer()
    moves = [
        {"from": "e2", "to": "e4", "uci": "e2e4"},
        {"from": "g1", "to": "f3", "uci": "g1f3"},
        {"from": "d2", "to": "d3", "uci": "d2d3"},
    ]
    chosen = pick_nn(_state(moves), model)
    assert chosen in moves


def test_returns_none_when_no_legal_moves():
    model = ChesskersScorer()
    assert pick_nn(_state([]), model) is None
    assert pick_nn({"fen": INITIAL_FEN}, model) is None  # missing key


def test_returns_singleton_when_only_one_move():
    model = ChesskersScorer()
    only = {"from": "e2", "to": "e4", "uci": "e2e4"}
    chosen = pick_nn(_state([only]), model)
    assert chosen is only


def test_picks_the_move_with_highest_logit():
    """Force a deterministic pick by mocking the model's forward output."""

    class StubModel(torch.nn.Module):
        def __init__(self, fixed_logits):
            super().__init__()
            self._logits = fixed_logits

        def eval(self):
            return self

        def forward(self, position, moves):
            return self._logits

    moves = [
        {"from": "a1", "to": "a2", "uci": "a1a2"},
        {"from": "b1", "to": "b2", "uci": "b1b2"},
        {"from": "c1", "to": "c2", "uci": "c1c2"},
    ]
    stub = StubModel(torch.tensor([0.1, 9.9, 0.5]))
    chosen = pick_nn(_state(moves), stub)
    assert chosen["uci"] == "b1b2"


def test_calls_model_in_eval_mode():
    """pick_nn should put the model in eval mode before inferring."""

    seen = []

    class TrackingModel(ChesskersScorer):
        def eval(self):
            seen.append("eval")
            return super().eval()

    model = TrackingModel()
    model.train()  # start in train mode
    moves = [{"from": "e2", "to": "e4", "uci": "e2e4"}]
    pick_nn(_state(moves), model)
    assert "eval" in seen
