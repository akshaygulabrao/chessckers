"""Load human-vs-engine games saved by the UI and convert them into
demonstration-bootstrapped training examples.

Each saved game in `games/games.jsonl` looks like:
  {
    "history": [{"fen": "...", "uci": "..."}, ...],  # one entry per ply
    "final_fen": "...",
    "final_status": "mate" | "variantEnd" | "stalemate" | None,
    "outcome": "white" | "black" | "draw" | "incomplete",
    "controllers": {"white": "...", "black": "..."},
    "saved_at": "...",
  }

`extract_examples(games, color, client)` returns a list of training examples
for the specified color's moves only:
  {"fen": str, "move": LegalMove dict, "target": float}
where target ∈ {+1, -1, 0} from `color`'s perspective (win / loss / draw).

The LegalMove dict comes from re-querying the API at each saved FEN and
matching the saved UCI; this is needed because encode_move() requires the
full LegalMove fields (from, to, waypoints, deployCount, etc.), and the UI
saves only the UCI string.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Protocol

from chessckers_engine.selfplay_az import AZExample

log = logging.getLogger("chessckers_engine.demos")

GameRecord = dict[str, Any]
Example = dict[str, Any]

_FEN_TURN = re.compile(r"^[^\s\[]+(?:\[[^\]]*\])?\s+([wb])\b")


class _Mover(Protocol):
    def new_game(self, fen: str | None = None) -> dict[str, Any]: ...


def load_games(path: str | Path) -> list[GameRecord]:
    """Read JSONL and return the parsed game records."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def filter_games_for_color(games: list[GameRecord], color: str) -> list[GameRecord]:
    """Keep only games where the specified color was human-played AND the game finished."""
    out = []
    for g in games:
        controllers = g.get("controllers") or {}
        if controllers.get(color) != "player":
            continue
        if g.get("outcome") in (None, "incomplete"):
            continue
        out.append(g)
    return out


def _turn_from_fen(fen: str) -> str:
    m = _FEN_TURN.match(fen)
    if not m:
        raise ValueError(f"unrecognized FEN: {fen!r}")
    return "white" if m.group(1) == "w" else "black"


def _target_for_color(outcome: str, color: str) -> float:
    if outcome == "draw":
        return 0.0
    return 1.0 if outcome == color else -1.0


def extract_examples(
    games: list[GameRecord],
    color: str,
    client: _Mover,
) -> list[Example]:
    """For each finished game where `color` was human-played, emit one
    `(fen, move, target)` example per move that color made, with `target` =
    +1/-1/0 from `color`'s perspective. Suitable for value-style scalar
    regression (`train.train()`).

    For policy imitation use `extract_az_examples` instead, which produces
    AZExamples with one-hot visit distributions on the played move."""
    examples: list[Example] = []
    for game in games:
        outcome = game["outcome"]
        target = _target_for_color(outcome, color)
        history = game.get("history") or []
        for entry in history:
            fen = entry["fen"]
            uci = entry["uci"]
            try:
                turn = _turn_from_fen(fen)
            except ValueError:
                continue
            if turn != color:
                continue
            try:
                state = client.new_game(fen)
            except Exception as e:  # noqa: BLE001
                log.debug("skipping fen=%s due to API error: %s", fen[:40], e)
                continue
            legal = state.get("legalMoves") or []
            match = next((m for m in legal if m.get("uci") == uci), None)
            if match is None:
                log.debug("skipping (fen, uci)=(%s, %s) — no matching legal move", fen[:40], uci)
                continue
            examples.append({"fen": fen, "move": match, "target": float(target)})
    return examples


def extract_az_examples(
    games: list[GameRecord],
    color: str,
    client: _Mover,
) -> list[AZExample]:
    """Imitation-learning examples in AZExample shape.

    For each move that `color` played:
    - `legal_moves` = full legal-moves list at that FEN (queried from the API).
    - `visit_distribution` = one-hot on the played move's index.
    - `wdl_target` = Win/Draw/Loss one-hot from `color`'s perspective; `moves_left_target` = plies to game end.

    Plug into `train_az.train_az()` to imitate the played moves (policy head,
    via cross-entropy with the one-hot target) AND learn position values from
    outcomes (value head, via MSE)."""
    examples: list[AZExample] = []
    for game in games:
        outcome = game["outcome"]
        target_v = _target_for_color(outcome, color)
        # WDL one-hot from color's POV (value target is now WDL, not a scalar).
        wdl = [1.0, 0.0, 0.0] if target_v > 0 else ([0.0, 0.0, 1.0] if target_v < 0 else [0.0, 1.0, 0.0])
        history = game.get("history") or []
        for j, entry in enumerate(history):
            fen = entry["fen"]
            uci = entry["uci"]
            try:
                turn = _turn_from_fen(fen)
            except ValueError:
                continue
            if turn != color:
                continue
            try:
                state = client.new_game(fen)
            except Exception as e:  # noqa: BLE001
                log.debug("skipping fen=%s due to API error: %s", fen[:40], e)
                continue
            legal = state.get("legalMoves") or []
            played_idx = next((i for i, m in enumerate(legal) if m.get("uci") == uci), None)
            if played_idx is None:
                log.debug("skipping (fen, uci)=(%s, %s) — no matching legal move", fen[:40], uci)
                continue
            dist = [0.0] * len(legal)
            dist[played_idx] = 1.0
            examples.append(
                AZExample(
                    fen=fen,
                    legal_moves=legal,
                    visit_distribution=dist,
                    wdl_target=wdl,
                    moves_left_target=float(len(history) - j),
                )
            )
    return examples
