import json

import pytest

from chessckers_engine.demos import (
    _target_for_color,
    _turn_from_fen,
    extract_examples,
    filter_games_for_color,
    load_games,
)

INITIAL_FEN_W = (
    "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"
    "[a6:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,"
    "a7:k,b7:k,c7:k,d7:k,e7:k,f7:k,g7:k,h7:k,"
    "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] w KQkq - 0 1"
)
FEN_BLACK_TO_MOVE = INITIAL_FEN_W.replace(" w ", " b ")


# ---- pure helpers ----


def test_turn_from_fen():
    assert _turn_from_fen(INITIAL_FEN_W) == "white"
    assert _turn_from_fen(FEN_BLACK_TO_MOVE) == "black"
    assert _turn_from_fen("8/8/8/8/8/8/8/8 b - - 0 1") == "black"


def test_turn_from_fen_invalid_raises():
    with pytest.raises(ValueError):
        _turn_from_fen("not a fen")


def test_target_for_color_win_loss_draw():
    assert _target_for_color("white", "white") == 1.0
    assert _target_for_color("white", "black") == -1.0
    assert _target_for_color("black", "black") == 1.0
    assert _target_for_color("black", "white") == -1.0
    assert _target_for_color("draw", "white") == 0.0
    assert _target_for_color("draw", "black") == 0.0


# ---- load + filter ----


def test_load_games_reads_jsonl(tmp_path):
    p = tmp_path / "games.jsonl"
    g1 = {"history": [], "outcome": "white", "controllers": {"white": "player", "black": "random"}}
    g2 = {"history": [], "outcome": "black", "controllers": {"white": "random", "black": "player"}}
    p.write_text(json.dumps(g1) + "\n" + json.dumps(g2) + "\n")
    loaded = load_games(p)
    assert len(loaded) == 2
    assert loaded[0]["outcome"] == "white"


def test_filter_games_keeps_only_player_color_finished():
    games = [
        {"controllers": {"white": "player", "black": "random"}, "outcome": "white"},
        {"controllers": {"white": "random", "black": "player"}, "outcome": "black"},
        {"controllers": {"white": "random", "black": "player"}, "outcome": "incomplete"},  # discarded
        {"controllers": {"white": "random", "black": "random"}, "outcome": "white"},        # discarded (no human)
    ]
    black_games = filter_games_for_color(games, "black")
    assert len(black_games) == 1
    assert black_games[0]["outcome"] == "black"


# ---- extract_examples ----


class _StubClient:
    """Stub Mover that returns canned legal moves per FEN."""

    def __init__(self, table: dict[str, list[dict]]) -> None:
        self.table = table
        self.calls: list[str] = []

    def new_game(self, fen=None):
        self.calls.append(fen or "")
        return {"fen": fen, "legalMoves": list(self.table.get(fen or "", []))}


def test_extract_examples_keeps_only_target_color_moves():
    move_w = {"uci": "e2e4", "from": "e2", "to": "e4"}
    move_b = {"uci": "f6e5", "from": "f6", "to": "e5"}
    client = _StubClient({INITIAL_FEN_W: [move_w], FEN_BLACK_TO_MOVE: [move_b]})

    games = [{
        "controllers": {"white": "random", "black": "player"},
        "outcome": "black",
        "history": [
            {"fen": INITIAL_FEN_W, "uci": "e2e4"},
            {"fen": FEN_BLACK_TO_MOVE, "uci": "f6e5"},
        ],
    }]
    examples = extract_examples(games, "black", client)
    # Only the black-turn move (f6e5) should be extracted.
    assert len(examples) == 1
    assert examples[0]["fen"] == FEN_BLACK_TO_MOVE
    assert examples[0]["move"] is move_b
    assert examples[0]["target"] == 1.0  # black won


def test_extract_examples_target_is_minus_one_when_color_lost():
    move_b = {"uci": "f6e5", "from": "f6", "to": "e5"}
    client = _StubClient({FEN_BLACK_TO_MOVE: [move_b]})

    games = [{
        "controllers": {"white": "random", "black": "player"},
        "outcome": "white",  # black lost
        "history": [{"fen": FEN_BLACK_TO_MOVE, "uci": "f6e5"}],
    }]
    examples = extract_examples(games, "black", client)
    assert examples[0]["target"] == -1.0


def test_extract_examples_target_is_zero_for_draws():
    move_b = {"uci": "f6e5", "from": "f6", "to": "e5"}
    client = _StubClient({FEN_BLACK_TO_MOVE: [move_b]})

    games = [{
        "controllers": {"white": "random", "black": "player"},
        "outcome": "draw",
        "history": [{"fen": FEN_BLACK_TO_MOVE, "uci": "f6e5"}],
    }]
    examples = extract_examples(games, "black", client)
    assert examples[0]["target"] == 0.0


def test_extract_examples_skips_uci_not_in_legal_moves():
    """If the saved UCI doesn't match any legal move at that FEN (e.g.,
    server replied differently), skip rather than crash."""
    available = {"uci": "e2e4", "from": "e2", "to": "e4"}
    client = _StubClient({INITIAL_FEN_W: [available]})

    games = [{
        "controllers": {"white": "player", "black": "random"},
        "outcome": "white",
        "history": [{"fen": INITIAL_FEN_W, "uci": "DIFFERENT_UCI"}],
    }]
    examples = extract_examples(games, "white", client)
    assert examples == []


def test_extract_examples_skips_invalid_fens():
    client = _StubClient({})
    games = [{
        "controllers": {"white": "player", "black": "random"},
        "outcome": "white",
        "history": [{"fen": "not-a-fen", "uci": "anything"}],
    }]
    assert extract_examples(games, "white", client) == []


def test_extract_examples_attaches_full_legalmove_dict():
    """The extracted example must reference the full LegalMove dict (with
    `from`, `to`, etc.), not just the UCI string. encode_move depends on this."""
    full_move = {
        "uci": "f6~g5~h4",
        "from": "f6",
        "to": "h4",
        "waypoints": ["g5", "h4"],
        "capture": "g5",
    }
    client = _StubClient({FEN_BLACK_TO_MOVE: [full_move]})
    games = [{
        "controllers": {"white": "random", "black": "player"},
        "outcome": "black",
        "history": [{"fen": FEN_BLACK_TO_MOVE, "uci": "f6~g5~h4"}],
    }]
    examples = extract_examples(games, "black", client)
    assert examples[0]["move"]["waypoints"] == ["g5", "h4"]
    assert examples[0]["move"]["capture"] == "g5"
