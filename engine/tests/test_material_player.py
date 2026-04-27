from typing import Any

from chessckers_engine.material_player import pick_material


class _StubClient:
    """Stub Mover that returns a pre-canned post-FEN per (input fen, uci) pair."""

    def __init__(self, table: dict[tuple[str, str], str]) -> None:
        self.table = table
        self.calls: list[tuple[str, str]] = []

    def make_move(self, fen: str, uci: str) -> dict[str, Any]:
        self.calls.append((fen, uci))
        return {"fen": self.table[(fen, uci)]}


# Position used as the "before" FEN; details don't matter to the test, only the
# post-FEN material does.
BEFORE_WHITE_TO_MOVE = (
    "8/8/8/8/8/8/PPPPPPPP/RNBQKBNR[a6:s] w - - 0 1"
)


def _post_fen(black_overlay: str, turn: str) -> str:
    """Construct a minimal post-move FEN with a given Black overlay."""
    return f"8/8/8/8/8/8/PPPPPPPP/RNBQKBNR[{black_overlay}] {turn} - - 0 1"


def test_returns_none_when_no_legal_moves():
    assert pick_material({"fen": "8/8/8/8/8/8/8/8 w - - 0 1", "legalMoves": []}, _StubClient({})) is None


def test_returns_singleton_when_only_one_move():
    only = {"uci": "e2e4"}
    state = {"fen": BEFORE_WHITE_TO_MOVE, "legalMoves": [only]}
    client = _StubClient({(BEFORE_WHITE_TO_MOVE, "e2e4"): _post_fen("a6:s", "b")})
    assert pick_material(state, client) is only


def test_picks_capture_over_quiet_move():
    """Two candidates: a quiet move that leaves Black material unchanged, and
    a capture that removes a Black Stone. Material picker must pick the capture."""
    quiet = {"uci": "e2e4"}
    capture = {"uci": "b1c3xa4"}  # uci string is opaque to the picker

    state = {"fen": BEFORE_WHITE_TO_MOVE, "legalMoves": [quiet, capture]}
    client = _StubClient(
        {
            (BEFORE_WHITE_TO_MOVE, quiet["uci"]): _post_fen("a6:s", "b"),       # 1 black stone left
            (BEFORE_WHITE_TO_MOVE, capture["uci"]): _post_fen("", "b"),         # zero black material
        }
    )
    assert pick_material(state, client) is capture


def test_picks_capture_of_higher_value_target():
    """Two captures: one removes a Black Stone (-1), one removes a Black King (-2).
    Picker must prefer the King capture."""
    take_stone = {"uci": "TAKES_STONE"}
    take_king = {"uci": "TAKES_KING"}

    state = {
        "fen": BEFORE_WHITE_TO_MOVE,
        "legalMoves": [take_stone, take_king],
    }
    client = _StubClient(
        {
            (BEFORE_WHITE_TO_MOVE, take_stone["uci"]): _post_fen("a7:k", "b"),  # king remains, stone gone
            (BEFORE_WHITE_TO_MOVE, take_king["uci"]): _post_fen("a6:s", "b"),   # stone remains, king gone
        }
    )
    assert pick_material(state, client) is take_king


def test_calls_make_move_once_per_legal_move():
    moves = [{"uci": f"M{i}"} for i in range(5)]
    state = {"fen": BEFORE_WHITE_TO_MOVE, "legalMoves": moves}
    client = _StubClient({(BEFORE_WHITE_TO_MOVE, m["uci"]): _post_fen("", "b") for m in moves})
    pick_material(state, client)
    assert len(client.calls) == 5


def test_score_is_from_mover_perspective_for_black():
    """Black's turn. After Black's move, side to move is White. The picker should
    pick the move whose post-FEN is *worst for White* (i.e. best for Black, because
    we want the player who just moved to maximize their own material)."""
    bad_for_black = {"uci": "BAD"}     # White ends up with king + 8 stones (lots)
    good_for_black = {"uci": "GOOD"}   # White ends up with no pawns

    BEFORE_BLACK = "8/8/8/8/8/8/PPPPPPPP/RNBQKBNR[a6:s] b - - 0 1"

    big_white_post = _post_fen("", "w")  # heavy White material, no Black
    small_white_post = "8/8/8/8/8/8/8/RNBQKBNR[] w - - 0 1"  # White minus its pawns

    client = _StubClient(
        {
            (BEFORE_BLACK, bad_for_black["uci"]): big_white_post,
            (BEFORE_BLACK, good_for_black["uci"]): small_white_post,
        }
    )
    state = {"fen": BEFORE_BLACK, "legalMoves": [bad_for_black, good_for_black]}
    assert pick_material(state, client) is good_for_black
