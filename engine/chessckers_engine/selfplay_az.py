"""AlphaZero-style self-play data generation for Chessckers.

At each move during a game we run PUCT MCTS, record the resulting visit
distribution, play the most-visited move (or sample from `visits**1/τ` if
exploration is desired), and continue. After the game ends, every recorded
position gets a value target equal to the eventual outcome from that
position's side-to-move perspective: +1 win, -1 loss, 0 draw.

Each AZExample produces, for one position visited during play:
- `fen`               — the position
- `legal_moves`       — the candidates considered at that position
- `visit_distribution`— normalized visit counts (a probability over
                        `legal_moves`) that becomes the policy target
- `value_target`      — outcome from STM's perspective (training target
                        for the value head)

These examples drop into the dual-loss training step in `train.py`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import torch

from chessckers_engine.mcts_puct import run_mcts
from chessckers_engine.model import ChesskersScorer

log = logging.getLogger("chessckers_engine.selfplay_az")

GameState = dict[str, Any]
LegalMove = dict[str, Any]


class _Mover(Protocol):
    def new_game(self, fen: str | None = None) -> GameState: ...
    def make_move(self, fen: str, uci: str) -> GameState: ...


class WatchSink(Protocol):
    """Receives per-move snapshots and per-game completion events for a
    spectator UI. `play_az_game` calls these synchronously."""

    def on_move(self, snapshot: dict[str, Any]) -> None: ...
    def on_game_end(self, game_log: dict[str, Any]) -> None: ...


class JsonlWatchSink:
    """Writes `current.json` (atomically replaced on every move) and appends
    finished games as one JSON line per game to `games.jsonl`. The viewer
    polls `current.json` to follow the live game and reads `games.jsonl` to
    let the user scrub through completed ones."""

    def __init__(self, watch_dir: Path):
        self.dir = Path(watch_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.current_path = self.dir / "current.json"
        self.games_path = self.dir / "games.jsonl"

    def on_move(self, snapshot: dict[str, Any]) -> None:
        tmp = self.current_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(snapshot))
        os.replace(tmp, self.current_path)  # atomic on POSIX

    def on_game_end(self, game_log: dict[str, Any]) -> None:
        game_log.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
        with self.games_path.open("a") as f:
            f.write(json.dumps(game_log))
            f.write("\n")


@dataclass
class AZRecord:
    fen: str
    legal_moves: list[LegalMove]
    visit_counts: list[int]   # aligned with legal_moves
    side_to_move: str          # "white" or "black"


@dataclass
class AZGame:
    records: list[AZRecord]
    final_status: str | None
    outcome: str  # "white" | "black" | "draw"


@dataclass
class AZExample:
    fen: str
    legal_moves: list[LegalMove]
    visit_distribution: list[float]  # probabilities, sum to ~1
    wdl_target: list[float]          # [P(win), P(draw), P(loss)] from STM POV — one-hot of the game outcome
    moves_left_target: float         # plies remaining from this position to game end


def _outcome_from_state(state: dict[str, Any]) -> str:
    """Authoritative outcome from a terminal game state.

    The server distinguishes Black's two win paths via `status` AND `winner`:
      - status='mate', winner='black'        → standard checkmate (chess-style)
      - status='variantEnd', winner='black'  → Black captured the White king
                                               directly via a chain/suicide move
    Both are Black wins. A previous version of this function keyed on
    `status` alone and treated 'variantEnd' as White-only — silently
    inverting value targets for every Black king-capture game.

    `winner` takes precedence; status-only fallback is for test stubs that
    don't simulate the full server response.
    """
    winner = state.get("winner")
    if winner == "white":
        return "white"
    if winner == "black":
        return "black"
    status = state.get("status")
    if status == "mate":
        return "black"
    if status == "variantEnd":
        return "white"
    return "draw"


def _aligned_visits(visit_dist: dict[str, int], legal_moves: list[LegalMove]) -> list[int]:
    return [visit_dist.get(m["uci"], 0) for m in legal_moves]


def _sample_move_index_from_visits(
    visits: list[int],
    temperature: float,
    rng: torch.Generator | None,
) -> int:
    """Sample an index into `visits` with probabilities ∝ visits**(1/τ).

    τ → 0 reduces to argmax; τ = 1.0 samples in proportion to visit counts."""
    if not visits:
        return 0
    if temperature <= 0:
        return int(max(range(len(visits)), key=lambda i: visits[i]))
    counts = torch.tensor(visits, dtype=torch.float32)
    if counts.sum() == 0:
        return 0
    probs = counts.pow(1.0 / temperature)
    probs = probs / probs.sum()
    return int(torch.multinomial(probs, num_samples=1, generator=rng).item())


def play_az_game(
    model: ChesskersScorer,
    client: _Mover,
    n_sims: int = 100,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    temp_cutoff_plies: int = 30,
    max_plies: int | None = None,
    rng: torch.Generator | None = None,
    dirichlet_alpha: float | None = 0.3,
    dirichlet_eps: float = 0.25,
    sink: WatchSink | None = None,
    sink_context: dict[str, Any] | None = None,
    vloss_batch: int = 1,
    search_fn: Any = None,
    resign_threshold: float = 0.0,
    resign_no_resign_frac: float = 0.1,
    resign_consecutive: int = 2,
    resign_min_ply: int = 8,
) -> AZGame:
    """Play one self-play game using PUCT MCTS at each move.

    `dirichlet_alpha=0.3` (AlphaZero-chess default) injects Dirichlet noise
    into root priors at every move, ensuring exploration of low-prior moves.
    Set to None to disable (e.g., for deterministic eval-style games).

    `temp_cutoff_plies` (AlphaZero behaviour): for the first `temp_cutoff_plies`
    plies the move is sampled at `temperature` (exploration / opening
    diversity); from then on it is played greedily (argmax of visit counts, i.e.
    τ→0). This keeps self-play games sharp after the opening so a won position
    is actually converted — a constant non-zero τ all game long lets the winner
    wander and the loser stall, which mislabels won positions as draws. Set very
    large to disable the cutoff (constant τ for the whole game).

    `sink` (optional) receives per-move snapshots and a per-game completion
    event so a spectator UI can follow training. `sink_context` is merged
    into every snapshot/log emitted (e.g. {iter, game_idx, total_games}).
    """
    if max_plies is None:
        # Env-overridable so experiments (e.g. a tiny endgame) can cap drawn
        # games short without threading a flag through every call site.
        max_plies = int(os.environ.get("CHESSCKERS_MAX_PLIES", "400"))
    state = client.new_game()
    records: list[AZRecord] = []
    history: list[dict[str, str]] = []  # [{fen, uci}], same schema as games.jsonl
    ctx = dict(sink_context or {})
    start_fen = state["fen"]
    if sink is not None:
        sink.on_move({**ctx, "ply": 0, "fen": start_fen, "history": [], "last_uci": None,
                      "temperature": temperature})

    ply = 0
    reuse_root = None  # tree reuse: the played move's subtree, carried to the next ply
    # Resignation (self-play throughput): end a game once the side-to-move's search
    # value is hopeless, so converted endgames don't burn plies past the decision.
    # `resign_threshold<=0` disables it. A held-out `resign_no_resign_frac` of games
    # never resign (play to a real terminal) so the false-positive rate stays
    # measurable — the lc0 calibration trick. Require the value to stay below
    # threshold for `resign_consecutive` plies (and past `resign_min_ply`) so a
    # single noisy eval can't end a game.
    resign_enabled = resign_threshold > 0.0
    no_resign_game = (not resign_enabled) or (
        rng is not None and float(torch.rand(1, generator=rng).item()) < resign_no_resign_frac
    )
    consec_resign = 0
    while not state.get("status") and ply < max_plies:
        legal = state.get("legalMoves") or []
        if not legal:
            break
        result = (search_fn or run_mcts)(
            state, client, model,
            n_sims=n_sims, c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            vloss_batch=vloss_batch,
            reuse_root=reuse_root,
        )
        visits = _aligned_visits(result.visit_distribution, legal)
        records.append(
            AZRecord(
                fen=state["fen"],
                legal_moves=legal,
                visit_counts=visits,
                side_to_move=state["turn"],
            )
        )
        # Resign check: the position is recorded above (a valid training example);
        # if the STM's search value is hopeless we end the game HERE, attributing
        # the win to the other side. `result.value` is the root Q (STM-relative);
        # it's None for search backends that don't expose it (resign just no-ops).
        if resign_enabled and not no_resign_game and ply >= resign_min_ply:
            v = getattr(result, "value", None)
            if v is not None and v <= -resign_threshold:
                consec_resign += 1
                if consec_resign >= resign_consecutive:
                    winner = "black" if state["turn"] == "white" else "white"
                    game = AZGame(records=records, final_status="resign", outcome=winner)
                    _emit_game_end(sink, ctx, history, state["fen"], game)
                    return game
            else:
                consec_resign = 0
        # AlphaZero per-ply temperature: explore the opening, then play sharp.
        eff_temp = temperature if ply < temp_cutoff_plies else 0.0
        if eff_temp <= 0:
            # Greedy: argmax visits, breaking ties by the SEARCH's own child
            # order (visit_distribution insertion order) rather than the
            # separately re-enumerated `legal` list, so the played move is
            # consistent with the tree that produced the visits — and identical
            # across search backends that share children but emit `legal` in a
            # different order (e.g. the native C++ loop, whose white move-gen
            # order differs from python-chess's). For the pure-Python search the
            # dict is built from `legal`, so this is the same move as before.
            best_uci = (max(result.visit_distribution.items(), key=lambda kv: kv[1])[0]
                        if result.visit_distribution else legal[0]["uci"])
            chosen = next(m for m in legal if m["uci"] == best_uci)
        else:
            idx = _sample_move_index_from_visits(visits, eff_temp, rng)
            chosen = legal[idx]
        prev_fen = state["fen"]
        history.append({"fen": prev_fen, "uci": chosen["uci"]})
        try:
            state = client.make_move(prev_fen, chosen["uci"])
        except Exception as e:  # noqa: BLE001
            log.debug("make_move failed at ply %d uci=%s: %s; ending game as draw", ply, chosen["uci"], e)
            game = AZGame(records=records, final_status=None, outcome="draw")
            _emit_game_end(sink, ctx, history, prev_fen, game)
            return game
        # Carry the played move's subtree into the next ply (re-rooted by run_mcts).
        reuse_root = result.root.children.get(chosen["uci"])
        ply += 1
        if sink is not None:
            sink.on_move({**ctx, "ply": ply, "fen": state["fen"], "history": list(history),
                          "last_uci": chosen["uci"], "temperature": temperature})

    status = state.get("status")
    game = AZGame(records=records, final_status=status, outcome=_outcome_from_state(state))
    _emit_game_end(sink, ctx, history, state["fen"], game)
    return game


def _emit_game_end(
    sink: WatchSink | None,
    ctx: dict[str, Any],
    history: list[dict[str, str]],
    final_fen: str,
    game: AZGame,
) -> None:
    if sink is None:
        return
    sink.on_game_end({
        **ctx,
        "history": history,
        "final_fen": final_fen,
        "final_status": game.final_status,
        "outcome": game.outcome,
        "controllers": {"white": "az", "black": "az"},
    })


def az_game_to_examples(game: AZGame, gamma: float | None = None) -> list[AZExample]:
    """Convert an AZGame to dual-target training examples.

    `gamma` is a per-ply value discount (env `CHESSCKERS_VALUE_DISCOUNT`,
    default 1.0 = off). The value target is scaled by `gamma**(plies-to-end)`,
    so the position one move before mate keeps the full ±1 while earlier
    positions decay toward 0. This gives the value head — and hence MCTS and
    the policy — an incentive to win *faster*: a mate-in-1 line scores higher
    than a mate-in-3, which flat ±1 targets cannot distinguish. (Symmetric:
    the losing side's target also decays, so it learns to delay the loss.)"""
    # WDL one-hot from the side-to-move's POV; win-SPEED is carried by the
    # moves-left target (plies-to-end), so the old gamma value-discount is gone.
    # `gamma` is accepted but ignored, for call-site compatibility.
    if game.outcome == "draw":
        wdl_white, wdl_black = [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]
    elif game.outcome == "white":
        wdl_white, wdl_black = [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]
    else:  # black
        wdl_white, wdl_black = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]

    out: list[AZExample] = []
    n = len(game.records)
    for i, rec in enumerate(game.records):
        total = sum(rec.visit_counts) or 1
        dist = [v / total for v in rec.visit_counts]
        wdl = wdl_white if rec.side_to_move == "white" else wdl_black
        out.append(
            AZExample(
                fen=rec.fen,
                legal_moves=rec.legal_moves,
                visit_distribution=dist,
                wdl_target=wdl,
                moves_left_target=float(n - i),  # plies remaining (record i -> game end)
            )
        )
    return out
