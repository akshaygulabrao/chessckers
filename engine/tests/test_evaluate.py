from chessckers_engine.evaluate import _status_to_outcome, evaluate, format_results, play_game


class _ScriptedClient:
    """Mover stub. `script` is a list of states the client returns in order;
    `new_game` returns states[0] and resets the cursor; `make_move` advances
    and returns the next state."""

    def __init__(self, states: list[dict]) -> None:
        self.states = states
        self.cursor = 0

    def new_game(self, fen=None):
        self.cursor = 0
        return self.states[0]

    def make_move(self, fen, uci):
        self.cursor += 1
        return self.states[self.cursor]


def _move(uci: str) -> dict:
    return {"uci": uci, "from": "a1", "to": "a2"}


def _state(turn: str, status: str | None = None, legal: list[dict] | None = None) -> dict:
    s = {"fen": "FEN", "turn": turn, "legalMoves": legal or [_move("M")]}
    if status:
        s["status"] = status
    return s


def test_status_to_outcome_mapping():
    assert _status_to_outcome("mate") == "black"
    assert _status_to_outcome("variantEnd") == "white"
    assert _status_to_outcome("stalemate") == "draw"
    assert _status_to_outcome(None) == "draw"
    assert _status_to_outcome("unexpected") == "draw"


def test_play_game_white_wins_when_variantEnd_status_appears():
    client = _ScriptedClient([
        _state("white"),
        _state("black", status="variantEnd"),
    ])
    picker = lambda s: s["legalMoves"][0]
    assert play_game(picker, picker, client) == "white"


def test_play_game_black_wins_on_mate():
    client = _ScriptedClient([
        _state("white"),
        _state("black"),
        _state("white", status="mate"),
    ])
    picker = lambda s: s["legalMoves"][0]
    assert play_game(picker, picker, client) == "black"


def test_play_game_draw_on_stalemate():
    client = _ScriptedClient([
        _state("white"),
        _state("black", status="stalemate"),
    ])
    picker = lambda s: s["legalMoves"][0]
    assert play_game(picker, picker, client) == "draw"


def test_play_game_draw_on_max_plies():
    """Pathological non-terminating client; max_plies bound forces a draw."""
    client = _ScriptedClient([_state("white") for _ in range(50)])
    picker = lambda s: s["legalMoves"][0]
    assert play_game(picker, picker, client, max_plies=5) == "draw"


def test_play_game_routes_picker_by_turn():
    """Different pickers produce distinguishable choices; verify each side's
    picker is consulted on its own turn."""
    seen: list[tuple[str, str]] = []

    def white(s):
        seen.append(("white-picker", s["turn"]))
        return s["legalMoves"][0]

    def black(s):
        seen.append(("black-picker", s["turn"]))
        return s["legalMoves"][0]

    client = _ScriptedClient([
        _state("white"),
        _state("black"),
        _state("white", status="variantEnd"),
    ])
    play_game(white, black, client)
    # White picker consulted on white's turn; black on black's turn.
    assert seen == [("white-picker", "white"), ("black-picker", "black")]


def test_evaluate_tallies_results_across_games():
    """Three rigged games: white wins, black wins, draw."""

    class MultiGameClient:
        def __init__(self, games: list[list[dict]]) -> None:
            self.games = list(games)
            self.current: list[dict] = []
            self.cursor = 0

        def new_game(self, fen=None):
            self.current = self.games.pop(0)
            self.cursor = 0
            return self.current[0]

        def make_move(self, fen, uci):
            self.cursor += 1
            return self.current[self.cursor]

    client = MultiGameClient([
        [_state("white"), _state("black", status="variantEnd")],            # white wins
        [_state("white"), _state("black"), _state("white", status="mate")], # black wins
        [_state("white"), _state("black", status="stalemate")],             # draw
    ])
    counts = evaluate(lambda s: s["legalMoves"][0], lambda s: s["legalMoves"][0], client, n_games=3)
    assert counts == {"white": 1, "black": 1, "draw": 1, "games": 3}


def test_format_results_shows_score_from_whites_perspective():
    msg = format_results("nn", "random", {"white": 3, "black": 1, "draw": 0, "games": 4})
    # 3 wins + 0 draws over 4 games → 0.750
    assert "0.750" in msg
    assert "nn (W) vs random (B)" in msg
