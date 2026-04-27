import random
from typing import Any

from chessckers_engine.dataset import generate_examples, load_jsonl, save_jsonl


class _ScriptedClient:
    """A Mover stub driven by a list of canned states.

    `new_game` returns states[0]. Each `make_move(fen, uci)` advances the
    pointer and returns the next state. Useful for stitching together a
    deterministic mini-game.
    """

    def __init__(self, states: list[dict[str, Any]]) -> None:
        self.states = states
        self.cursor = 0
        self.move_calls: list[tuple[str, str]] = []

    def new_game(self, fen: str | None = None) -> dict[str, Any]:
        self.cursor = 0
        return self.states[0]

    def make_move(self, fen: str, uci: str) -> dict[str, Any]:
        self.move_calls.append((fen, uci))
        self.cursor += 1
        if self.cursor >= len(self.states):
            return {"fen": "8/8/8/8/8/8/8/8 w - - 0 1", "status": "variantEnd", "legalMoves": []}
        return self.states[self.cursor]


# Tiny non-Chessckers-realistic FENs used purely as opaque keys (the dataset
# code only parses them via material_for_side_to_move, which we control via
# overlay shape).
def _post_fen_white_to_move(black_overlay: str) -> str:
    return f"8/8/8/8/8/8/PPPPPPPP/RNBQKBNR[{black_overlay}] w - - 0 1"


def _post_fen_black_to_move(black_overlay: str) -> str:
    return f"8/8/8/8/8/8/PPPPPPPP/RNBQKBNR[{black_overlay}] b - - 0 1"


def test_empty_state_yields_no_examples():
    client = _ScriptedClient([{"fen": "F0", "status": "variantEnd", "legalMoves": []}])
    assert generate_examples(1, client) == []


def test_one_position_yields_one_example_per_legal_move():
    move_a = {"uci": "M1", "from": "e2", "to": "e4"}
    move_b = {"uci": "M2", "from": "g1", "to": "f3"}
    move_c = {"uci": "M3", "from": "d2", "to": "d4"}
    states = [
        {"fen": "F0", "turn": "white", "legalMoves": [move_a, move_b, move_c]},
        # post-move states for each of the 3 fanned-out moves
        {"fen": _post_fen_black_to_move("a6:s"), "legalMoves": []},
        {"fen": _post_fen_black_to_move("a6:s"), "legalMoves": []},
        {"fen": _post_fen_black_to_move("a6:s"), "legalMoves": []},
    ]

    # Custom client: make_move returns post states cyclically by call index;
    # but the third make_move (the one that "advances" the game) must yield
    # status to terminate.
    class FanoutClient:
        def __init__(self) -> None:
            self.move_calls = 0

        def new_game(self, fen=None):
            return states[0]

        def make_move(self, fen, uci):
            self.move_calls += 1
            # First 3 calls: scoring fan-out. 4th call: advance ply 1 → end.
            if self.move_calls <= 3:
                return states[1]
            return {"fen": "FEND", "status": "variantEnd", "legalMoves": []}

    client = FanoutClient()
    rng = random.Random(0)
    examples = generate_examples(1, client, rng=rng)
    assert len(examples) == 3
    assert {ex["move"]["uci"] for ex in examples} == {"M1", "M2", "M3"}
    assert all(ex["fen"] == "F0" for ex in examples)


def test_target_sign_for_white_capture_of_black_stone():
    """White moves; resulting position has zero Black material vs starting one Stone.
    Target (from White's perspective) should be positive."""

    capture = {"uci": "CAPTURE", "from": "a3", "to": "a6"}
    states = [
        {"fen": "F0", "turn": "white", "legalMoves": [capture]},
        # post: it's Black to move; Black has nothing → black_material=0; raw=W-B=W
        {"fen": _post_fen_black_to_move(""), "legalMoves": []},
    ]

    class C:
        def __init__(self) -> None:
            self.calls = 0

        def new_game(self, fen=None):
            return states[0]

        def make_move(self, fen, uci):
            self.calls += 1
            if self.calls == 1:
                return states[1]
            return {"fen": "FEND", "status": "variantEnd", "legalMoves": []}

    examples = generate_examples(1, C())
    assert len(examples) == 1
    # Target is from the moving side's perspective. White captured; Black has 0
    # material in the resulting position; White has its standard pawns + back rank
    # with king_value=0 -> 8 + 6 + 6 + 10 + 9 = 39. Black=0. Side-to-move=Black,
    # so material_for_side_to_move = -(39-0) = -39. Negated for moving side: +39.
    assert examples[0]["target"] == 39.0


def test_max_plies_bound_terminates_game():
    """A pathological never-ending stub should terminate at max_plies."""

    move = {"uci": "M", "from": "a1", "to": "a2"}

    valid_fen = "8/8/8/8/8/8/8/8 w - - 0 1"

    class EternalClient:
        def __init__(self) -> None:
            self.calls = 0

        def new_game(self, fen=None):
            return {"fen": valid_fen, "legalMoves": [move]}

        def make_move(self, fen, uci):
            self.calls += 1
            return {"fen": valid_fen, "legalMoves": [move]}

    examples = generate_examples(1, EternalClient(), max_plies=5)
    # 5 plies × 1 fan-out + 5 advance moves = 10 make_move calls
    # Examples are 1 per ply (only one legal move at each)
    assert len(examples) == 5


def test_save_and_load_jsonl_roundtrip(tmp_path):
    path = tmp_path / "dataset.jsonl"
    examples = [
        {"fen": "F0", "move": {"uci": "M1"}, "target": 1.5},
        {"fen": "F1", "move": {"uci": "M2"}, "target": -3.0},
    ]
    n = save_jsonl(examples, path)
    assert n == 2
    loaded = load_jsonl(path)
    assert loaded == examples


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "with_blanks.jsonl"
    path.write_text('{"fen":"F","move":{"uci":"M"},"target":0.0}\n\n\n')
    assert len(load_jsonl(path)) == 1
