import torch

from chessckers_engine.model import ChesskersScorer
from chessckers_engine.selfplay import (
    Decision,
    SelfPlayGame,
    decisions_to_examples,
    play_self_game,
    sample_move,
)


INITIAL_FEN = (
    "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"
    "[a6:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,"
    "a7:k,b7:k,c7:k,d7:k,e7:k,f7:k,g7:k,h7:k,"
    "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] w KQkq - 0 1"
)


# --- sample_move ---


def test_sample_move_returns_none_on_empty_legal_moves():
    model = ChesskersScorer()
    state = {"fen": INITIAL_FEN, "legalMoves": []}
    assert sample_move(model, state, rng=None, temperature=1.0) is None


def test_sample_move_returns_a_legal_move_at_temperature_one():
    torch.manual_seed(0)
    model = ChesskersScorer()
    moves = [
        {"from": "e2", "to": "e4", "uci": "e2e4"},
        {"from": "g1", "to": "f3", "uci": "g1f3"},
    ]
    state = {"fen": INITIAL_FEN, "legalMoves": moves}
    chosen = sample_move(model, state, rng=None, temperature=1.0)
    assert chosen in moves


def test_sample_move_at_temperature_zero_is_argmax():
    """With τ=0 the sampler should pick the argmax move deterministically."""
    torch.manual_seed(0)
    model = ChesskersScorer()
    moves = [
        {"from": "a2", "to": "a3", "uci": "a2a3"},
        {"from": "b2", "to": "b3", "uci": "b2b3"},
        {"from": "c2", "to": "c3", "uci": "c2c3"},
    ]
    state = {"fen": INITIAL_FEN, "legalMoves": moves}
    # Multiple samples should all be the same.
    picks = {sample_move(model, state, rng=None, temperature=0.0)["uci"] for _ in range(10)}
    assert len(picks) == 1


def test_sample_move_at_high_temperature_is_diverse():
    """With τ very high, sampling should approach uniform — multiple draws should
    cover all candidates eventually."""
    torch.manual_seed(0)
    model = ChesskersScorer()
    moves = [{"uci": f"M{i}", "from": "a1", "to": "a2"} for i in range(8)]
    state = {"fen": INITIAL_FEN, "legalMoves": moves}
    g = torch.Generator().manual_seed(0)
    seen = {sample_move(model, state, rng=g, temperature=100.0)["uci"] for _ in range(200)}
    # With τ=100 over 8 moves, all should appear with high probability across 200 samples.
    assert len(seen) == 8


# --- play_self_game ---


class _ScriptedClient:
    """Stub Mover that returns a pre-canned sequence of states."""

    def __init__(self, states: list[dict]) -> None:
        self.states = states
        self.cursor = 0

    def new_game(self, fen=None):
        self.cursor = 0
        return self.states[0]

    def make_move(self, fen, uci):
        self.cursor += 1
        return self.states[self.cursor]


def _state(turn: str, status: str | None = None, legal: list[dict] | None = None) -> dict:
    base_legal = legal or [{"from": "e2", "to": "e4", "uci": "e2e4"}]
    s = {"fen": INITIAL_FEN, "turn": turn, "legalMoves": base_legal}
    if status:
        s["status"] = status
    return s


def test_play_self_game_records_one_decision_per_ply():
    torch.manual_seed(0)
    model = ChesskersScorer()
    client = _ScriptedClient([
        _state("white"),
        _state("black"),
        _state("white", status="variantEnd"),
    ])
    game = play_self_game(model, client, temperature=1.0)
    assert len(game.decisions) == 2  # white played, then black played, then status set
    assert game.decisions[0].side_to_move == "white"
    assert game.decisions[1].side_to_move == "black"
    assert game.outcome == "white"
    assert game.final_status == "variantEnd"


def test_play_self_game_handles_mate_status():
    torch.manual_seed(0)
    model = ChesskersScorer()
    client = _ScriptedClient([
        _state("white"),
        _state("black", status="mate"),
    ])
    game = play_self_game(model, client, temperature=1.0)
    assert game.outcome == "black"


def test_play_self_game_max_plies_yields_draw_with_no_status():
    torch.manual_seed(0)
    model = ChesskersScorer()
    # Infinite-loop client: every state has no status and offers one move.
    class Eternal:
        def __init__(self) -> None:
            self.count = 0

        def new_game(self, fen=None):
            return _state("white")

        def make_move(self, fen, uci):
            self.count += 1
            return _state("black" if self.count % 2 else "white")

    game = play_self_game(model, Eternal(), temperature=1.0, max_plies=5)
    assert len(game.decisions) == 5
    assert game.final_status is None
    assert game.outcome == "draw"


# --- decisions_to_examples ---


def test_decisions_to_examples_assigns_plus_one_to_winning_side():
    decisions = [
        Decision(fen="F1", move={"uci": "M1"}, side_to_move="white"),
        Decision(fen="F2", move={"uci": "M2"}, side_to_move="black"),
        Decision(fen="F3", move={"uci": "M3"}, side_to_move="white"),
    ]
    game = SelfPlayGame(decisions=decisions, final_status="variantEnd", outcome="white")
    examples = decisions_to_examples(game)
    assert [ex["target"] for ex in examples] == [1.0, -1.0, 1.0]


def test_decisions_to_examples_zeros_targets_for_draw():
    decisions = [Decision(fen="F", move={"uci": "M"}, side_to_move="white")]
    game = SelfPlayGame(decisions=decisions, final_status="stalemate", outcome="draw")
    assert decisions_to_examples(game) == [{"fen": "F", "move": {"uci": "M"}, "target": 0.0}]


def test_decisions_to_examples_assigns_minus_one_to_losing_side_in_black_win():
    decisions = [
        Decision(fen="F1", move={"uci": "M1"}, side_to_move="white"),
        Decision(fen="F2", move={"uci": "M2"}, side_to_move="black"),
    ]
    game = SelfPlayGame(decisions=decisions, final_status="mate", outcome="black")
    examples = decisions_to_examples(game)
    assert [ex["target"] for ex in examples] == [-1.0, 1.0]
