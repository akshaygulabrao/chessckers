import torch

from chessckers_engine.model import ChesskersScorer
from chessckers_engine.selfplay_az import (
    AZExample,
    AZGame,
    AZRecord,
    _aligned_visits,
    _outcome_from_state,
    _sample_move_index_from_visits,
    az_game_to_examples,
    play_az_game,
)


def _move(uci: str) -> dict:
    return {"uci": uci, "from": "a1", "to": "a2"}


# ---- visit alignment ----


def test_aligned_visits_orders_by_legal_moves_uci():
    legal = [_move("A"), _move("B"), _move("C")]
    dist = {"B": 5, "A": 1, "C": 0}
    assert _aligned_visits(dist, legal) == [1, 5, 0]


def test_aligned_visits_returns_zero_for_missing_uci():
    legal = [_move("A"), _move("B")]
    dist = {"A": 3}
    assert _aligned_visits(dist, legal) == [3, 0]


# ---- visit-count sampling ----


def test_sample_at_temperature_zero_picks_argmax():
    visits = [1, 5, 3]
    assert _sample_move_index_from_visits(visits, temperature=0.0, rng=None) == 1


def test_sample_at_temperature_one_distribution_matches_visits():
    """Over many samples, frequency of picking each index should track
    visit-count proportions (within statistical noise)."""
    visits = [1, 9]  # 10% / 90%
    g = torch.Generator().manual_seed(0)
    counts = [0, 0]
    for _ in range(2000):
        counts[_sample_move_index_from_visits(visits, temperature=1.0, rng=g)] += 1
    # Expect ~200 vs ~1800; tolerate noise.
    assert 100 < counts[0] < 350
    assert 1650 < counts[1] < 1900


def test_sample_with_zero_total_visits_returns_first_index():
    assert _sample_move_index_from_visits([0, 0, 0], temperature=1.0, rng=None) == 0


# ---- play_az_game with stub client ----


_FEN_W = "8/8/8/8/8/8/8/4K3 w - - 0 1"
_FEN_B = "8/8/8/8/8/8/8/4K3 b - - 0 1"


def _state(turn: str, status: str | None = None, legal: list[dict] | None = None) -> dict:
    s = {"fen": _FEN_W if turn == "white" else _FEN_B, "turn": turn, "legalMoves": legal or [_move("M")]}
    if status:
        s["status"] = status
    return s


class _GameClient:
    """A stub that tracks the game cursor only on calls whose `fen` matches
    the current state's fen — i.e., the game-advancing calls made by
    play_az_game's outer loop. MCTS's internal make_move calls (which use the
    same fens too, hence indistinguishable from the outside) are intercepted
    by monkeypatching run_mcts itself."""

    def __init__(self, states: list[dict]) -> None:
        self.states = states
        self.cursor = 0

    def new_game(self, fen=None):
        self.cursor = 0
        return self.states[0]

    def make_move(self, fen, uci):
        self.cursor += 1
        if self.cursor >= len(self.states):
            return {"fen": _FEN_W, "status": "variantEnd", "legalMoves": []}
        return self.states[self.cursor]


def _fake_run_mcts_returning_uniform_visits(state, client, model, **_kwargs):
    """Stand-in for run_mcts that distributes visits uniformly across legal moves.
    Accepts and ignores all kwargs (n_sims, c_puct, dirichlet_alpha, dirichlet_eps)."""
    from chessckers_engine.mcts_puct import MctsResult, PuctNode

    legal = state.get("legalMoves") or []
    n_sims = _kwargs.get("n_sims", 8)
    visit_dist = {m["uci"]: max(n_sims // max(len(legal), 1), 1) for m in legal}
    chosen = legal[0] if legal else None
    return MctsResult(chosen=chosen, visit_distribution=visit_dist, root=PuctNode(fen=state["fen"], move_to_here=None))


def test_outcome_from_state_uses_winner_field():
    """Regression: status='variantEnd' with winner='black' is the Black
    king-capture path. Keying on status alone inverts the value target
    for every such game — a real bug that silently corrupted training."""
    assert _outcome_from_state({"status": "variantEnd", "winner": "black"}) == "black"
    assert _outcome_from_state({"status": "variantEnd", "winner": "white"}) == "white"
    assert _outcome_from_state({"status": "mate", "winner": "black"}) == "black"
    # Status fallback when winner is missing (test-stub compatibility).
    assert _outcome_from_state({"status": "variantEnd"}) == "white"
    assert _outcome_from_state({"status": "mate"}) == "black"
    assert _outcome_from_state({}) == "draw"


def test_play_az_game_records_one_per_ply_and_outcome(monkeypatch):
    monkeypatch.setattr("chessckers_engine.selfplay_az.run_mcts", _fake_run_mcts_returning_uniform_visits)
    torch.manual_seed(0)
    model = ChesskersScorer()
    client = _GameClient([
        _state("white"),
        _state("black"),
        _state("white", status="variantEnd"),
    ])
    game = play_az_game(model, client, n_sims=4, temperature=1.0)
    assert game.outcome == "white"
    assert len(game.records) == 2
    assert game.records[0].side_to_move == "white"
    assert game.records[1].side_to_move == "black"


def test_play_az_game_records_visit_counts_aligned_to_legal_moves(monkeypatch):
    monkeypatch.setattr("chessckers_engine.selfplay_az.run_mcts", _fake_run_mcts_returning_uniform_visits)
    torch.manual_seed(0)
    model = ChesskersScorer()
    client = _GameClient([
        _state("white", legal=[_move("M1"), _move("M2"), _move("M3")]),
        _state("black", status="mate"),
    ])
    game = play_az_game(model, client, n_sims=8, temperature=1.0)
    rec = game.records[0]
    assert len(rec.visit_counts) == 3  # one count per legal move
    assert sum(rec.visit_counts) > 0


# ---- examples conversion ----


def test_examples_normalize_visit_distribution_to_probabilities():
    rec = AZRecord(
        fen="F", legal_moves=[_move("A"), _move("B"), _move("C")],
        visit_counts=[2, 6, 0], side_to_move="white",
    )
    game = AZGame(records=[rec], final_status="variantEnd", outcome="white")
    [ex] = az_game_to_examples(game)
    assert ex.visit_distribution == [2 / 8, 6 / 8, 0.0]
    assert abs(sum(ex.visit_distribution) - 1.0) < 1e-9


def test_examples_wdl_target_matches_outcome_from_movers_perspective():
    white_rec = AZRecord(fen="F1", legal_moves=[_move("M")], visit_counts=[1], side_to_move="white")
    black_rec = AZRecord(fen="F2", legal_moves=[_move("M")], visit_counts=[1], side_to_move="black")
    WIN, DRAW, LOSS = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]

    # white wins: WDL one-hot from each mover's POV.
    game_w = AZGame(records=[white_rec, black_rec], final_status="variantEnd", outcome="white")
    exs = az_game_to_examples(game_w)
    assert exs[0].wdl_target == WIN    # white moved here, white won
    assert exs[1].wdl_target == LOSS   # black moved here, black lost
    # moves-left = plies remaining (record i of n -> n - i)
    assert exs[0].moves_left_target == 2.0
    assert exs[1].moves_left_target == 1.0

    # black wins
    game_b = AZGame(records=[white_rec, black_rec], final_status="mate", outcome="black")
    exs = az_game_to_examples(game_b)
    assert exs[0].wdl_target == LOSS
    assert exs[1].wdl_target == WIN

    # draw
    game_d = AZGame(records=[white_rec], final_status="stalemate", outcome="draw")
    exs = az_game_to_examples(game_d)
    assert exs[0].wdl_target == DRAW


def test_examples_handles_zero_visit_record_without_dividing_by_zero():
    rec = AZRecord(
        fen="F", legal_moves=[_move("A")], visit_counts=[0], side_to_move="white",
    )
    game = AZGame(records=[rec], final_status="stalemate", outcome="draw")
    [ex] = az_game_to_examples(game)
    # Sum-zero visits become a degenerate distribution of [0.0]; not crashing is the win.
    assert ex.visit_distribution == [0.0]
    assert ex.wdl_target == [0.0, 1.0, 0.0]  # draw
