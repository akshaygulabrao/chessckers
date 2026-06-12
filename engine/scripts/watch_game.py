#!/usr/bin/env python
"""Watch a self-play game from any Chessckers FEN.

The trained net plays BOTH sides at --sims (default 200, matching self-play) MCTS sims/move. The
MOVE played is always the argmax of the visit counts (the "calculation");
exploration is injected only as root Dirichlet noise (--explore, default
0.30 = 30%), so different runs (or --seed) give varied games while each move
stays the search's best. --explore 0 = pure greedy/deterministic. Each ply
renders the 10x10 board live, then the net's raw WDL eval + moves-left estimate
(both from the mover's POV), then the engine's top 3 lines for that position:
MCTS policy probability (visits), value (mover's POV), and the principal
variation. Defaults to the latest ckpt.

This is THE tool for inspecting a self-play game — whether watching the net play
live, replaying a PGN movelist, or replaying a RECORDED game (a ccz1 chunk from
the server / DB). Reach for --chunk instead of hand-decoding chunks: the move a
game actually played is SAMPLED from the visit counts, so it is NOT the visit-
argmax, and only this path (FEN-matched reconstruction) renders the real game.

  cd engine
  .venv/bin/python scripts/watch_game.py "8/8/8/8/3kk3/8/8/4K3[d4:kk,e4:kk] b - - 0 1"   # net plays
  .venv/bin/python scripts/watch_game.py --moves "<selfplay PGN line>" --no-eval         # replay a movelist
  .venv/bin/python scripts/watch_game.py --chunk ../lczero-server/games/run1/training.1079.gz  # replay a DB game
  # options: --weights X.pt  --sims 200  --max-plies 200  --device cpu|mps  --delay 0.5
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

_ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../engine

TOP_N = 3       # candidate moves ("lines") to show per position
PV_MAX = 8      # plies of principal-variation continuation to show per line

# The simplified training start (White's 8 pawns + king vs three 2-King towers on
# d6/e6/f6) — the default replay/watch position; mirrors the engine's kStartposFen.
# Fallback only — keep loosely in sync with the fork's kStartposFen.
_FALLBACK_START_FEN = "3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] b - - 0 1"


def _engine_start_fen() -> str:
    """The start position the fleet actually trains from = the fork's kStartposFen
    (akshay-chessckers-0/src/chess/board.cc). Read it from source so this default can
    never drift from the engine (it concatenates adjacent C++ string literals); fall
    back to the constant if the file is unreadable."""
    board = os.path.join(_ENG, "..", "akshay-chessckers-0", "src", "chess", "board.cc")
    try:
        m = re.search(r"kStartposFen\s*=\s*(.+?);", open(board).read(), re.S)
        if m:
            fen = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))).strip()
            if fen:
                return fen
    except OSError:
        pass
    return _FALLBACK_START_FEN


DEFAULT_START_FEN = _engine_start_fen()

_RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}


def _parse_movelist(s: str) -> list[str]:
    """Tokens from a movelist / raw selfplay PGN line. Drops a leading 'PGN:', and
    stops at the game result ('1-0' etc.) or any trailing '{...}' comment (e.g.
    '{OL: 0}'). Inline move annotations like 'f6g6{1}' / 'd6c5[1]' are kept — they
    are part of the Chessckers UCI and match PyVariant's emitted uci exactly."""
    s = s.strip()
    if s.startswith("PGN:"):
        s = s[len("PGN:"):]
    out: list[str] = []
    for tok in s.split():
        if tok in _RESULT_TOKENS or tok.startswith("{"):
            break
        out.append(tok)
    return out


def _resolve_weights(arg: str, latest: bool = False) -> list[str]:
    """Ordered checkpoint candidates, freshest first. The async/continuous
    trainer (train_continuous.py) writes weights/run/{weights.pt, iter-async-
    NNNN.pt}; weights.pt is the live rolling publish (newest but may be mid-
    write), so it leads and the numbered durable snapshots follow as fallbacks.
    base_wdl_v* are the WDL-arch bases. (The pre-WDL base_live.pt / iter-az-*.pt
    are deliberately excluded — old scalar value head, won't load here.)"""
    if arg:
        return [arg]
    run = os.path.join(_ENG, "weights/run")
    numbered = glob.glob(os.path.join(run, "iter-async-[0-9]*.pt"))
    iters = sorted(
        numbered,
        key=lambda p: int(re.search(r"(\d+)\.pt$", p).group(1)),
        reverse=True,
    )
    cands = [
        os.path.join(run, "weights.pt"),
        *iters,
        os.path.join(_ENG, "weights/base_wdl_v2.pt"),
        os.path.join(_ENG, "weights/base_wdl_v1.pt"),
    ]
    if latest:
        # The live published net the fleet is training right now: the trainer's
        # rolling weights.pt in the server run-dir. Leads so --latest picks it first.
        cands.insert(0, os.path.join(_ENG, "..", "lczero-server",
                                     "trainer", "run1", "weights.pt"))
    found = [p for p in cands if os.path.exists(p)]
    if not found:
        raise SystemExit("no WDL weights found (weights/run/weights.pt, iter-async-*.pt, "
                         "or weights/base_wdl_v*.pt); pass --weights")
    return found


def _pv_ucis(child, max_len: int = PV_MAX) -> list[str]:
    """Principal variation starting at a root child: descend by most-visited
    child each ply. Returns [child's move, best reply, ...]. Stops at a
    terminal/unexpanded node or an unexplored (0-visit) child."""
    ucis: list[str] = []
    node = child
    for _ in range(max_len):
        if node is None or node.move_to_here is None:
            break
        ucis.append(node.move_to_here["uci"])
        if node.is_terminal or not node.children:
            break
        nxt = max(node.children.values(), key=lambda c: c.visits)
        if nxt.visits == 0:
            break
        node = nxt
    return ucis


def _print_net_eval(model, fen: str, turn: str, device: str) -> None:
    """Show the value head's raw WDL distribution and the moves-left head's
    plies-to-end estimate for this position (both from the mover's POV) — the
    network's un-searched read, distinct from the MCTS-backed `ev` per move."""
    import torch

    from chessckers_engine.encoding import encoders_for

    enc_pos, _, _ = encoders_for(getattr(model, "VERSION", "v1"))
    pos = enc_pos(fen).unsqueeze(0).to(device)
    with torch.no_grad():
        # V1 pools the trunk to a vector internally; V2/V3 keep the trunk spatial, so pool it
        # here (mean over the 10x10) before the shared value/moves-left heads.
        emb = (model._pooled(model._spatial(pos)) if getattr(model, "VERSION", "v1") == "v2"
               else model._position_embedding(pos))
        wdl = torch.softmax(model.value_head(emb), dim=-1).reshape(-1)
        mlh = float(model.moves_left_head(emb).reshape(()).item())
    w, d, lose = (100.0 * wdl[0].item(), 100.0 * wdl[1].item(), 100.0 * wdl[2].item())
    print(f"  {turn} net eval: W {w:4.1f}% / D {d:4.1f}% / L {lose:4.1f}%  moves-left ~{mlh:.0f}")


def _print_analysis(turn: str, result, n_sims: int, top_n: int = TOP_N) -> None:
    """Show the top-N moves the search considered at this position: MCTS policy
    probability (proportional to visits), value from the mover's perspective
    (-childQ), visit count, and the principal-variation continuation."""
    children = sorted(result.root.children.values(), key=lambda c: c.visits, reverse=True)
    if not children:
        return
    total = sum(c.visits for c in children) or 1
    print(f"  {turn} to move — top {min(top_n, len(children))} of {len(children)} ({n_sims} sims):")
    for i, c in enumerate(children[:top_n], 1):
        uci = c.move_to_here["uci"] if c.move_to_here else "?"
        pct = 100.0 * c.visits / total
        ev = -c.q or 0.0  # child.q is from the child's POV; negate -> mover's POV (avoid -0.00)
        pv = _pv_ucis(c)
        cont = " ".join(pv[1:]) if len(pv) > 1 else ("# mate" if c.is_terminal else "")
        print(f"    {i}. {uci:<14} {pct:5.1f}%  ev {ev:+.2f}  n={c.visits:<4} {cont}")


def _replay(args, client, state, model, show_board, outcome_from_state) -> int:
    """Replay --moves token-by-token from `state`, rendering each ply (and the net's
    WDL eval unless --no-eval). Each token is fed straight to make_move: PyVariant's
    emitted uci matches the selfplay PGN tokens exactly (cadence/deploy/sub-move)."""
    toks = _parse_movelist(args.moves)
    if not toks:
        raise SystemExit("--moves had no playable tokens")
    print(f"replaying {len(toks)} plies"
          + ("" if args.no_eval else f"  (net WDL each ply; weights loaded)") + "\n")
    for ply, tok in enumerate(toks, 1):
        if args.clear:
            _clear_screen()
        if model is not None:
            _print_net_eval(model, state["fen"], state["turn"], args.device)  # mover-POV WDL
        try:
            state = client.make_move(state["fen"], tok)
        except Exception as e:  # noqa: BLE001 — surface the bad token + legal set to debug
            print(f"\n!! ply {ply}: could not apply token {tok!r}: {e}")
            legal = [m["uci"] for m in (state.get("legalMoves") or [])]
            print(f"   {len(legal)} legal here: {legal[:40]}")
            return 1
        show_board(ply, tok, state["fen"])
        if state.get("status"):
            break
        if not _advance(args):
            break
    status = state.get("status")
    plies = ply
    if status:
        outcome = outcome_from_state(state)
        print(f"\n######## {outcome.upper()} WINS ({status}) in {plies} plies ########"
              if outcome != "draw" else f"\n######## DRAW ({status}) in {plies} plies ########")
    else:
        print(f"\n######## replay ended at {plies} plies — game not terminal here ########")
    return 0


def _clear_screen() -> None:
    """Home the cursor and clear, so each ply renders in place instead of scrolling."""
    print("\033[H\033[2J", end="", flush=True)


def _read_key() -> str:
    """Block for ONE keypress (raw mode) and return it. Falls back to a line read
    (Enter) when stdin isn't a TTY (e.g. piped), so non-interactive runs don't hang."""
    if not sys.stdin.isatty():
        sys.stdin.readline()
        return "\n"
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _advance(args) -> bool:
    """Pause between plies. --step waits for a key (space/Enter advance; q or Ctrl-C/D
    quits); otherwise --delay sleeps. Returns False if the user asked to quit."""
    if args.step:
        return _read_key() not in ("q", "Q", "\x03", "\x04")
    if args.delay:
        time.sleep(args.delay)
    return True


def _moves_from_chunk(path: str) -> tuple[str, list[str]]:
    """Recover (start FEN, actual move line) from a recorded ccz1 self-play chunk
    (e.g. lczero-server/games/run1/training.N.gz). The chunk stores one position
    per ply but NOT the move played — and the played move was SAMPLED from the
    visit counts, so it is NOT the visit-argmax. We recover each move by finding
    the legal move whose result matches the NEXT ply's FEN; the final ply (-> the
    terminal/mate, which has no next FEN) falls back to the visit-argmax. This is
    why --chunk exists: hand-rolling decode + argmax silently mislabels the game."""
    from chessckers_engine.training_chunk import decode_chunk
    from chessckers_engine.variant_py import PyVariantClient

    exs = decode_chunk(open(path, "rb").read())
    if not exs:
        raise SystemExit(f"--chunk {path}: empty or undecodable chunk")
    client = PyVariantClient()

    def applied_fen(fen: str, uci: str) -> str | None:
        try:
            return client.make_move(fen, uci)["fen"]
        except Exception:  # noqa: BLE001 — illegal/garbled token, just skip it
            return None

    moves: list[str] = []
    for i in range(len(exs) - 1):
        nxt = exs[i + 1].fen
        played = next((m["uci"] for m in exs[i].legal_moves
                       if applied_fen(exs[i].fen, m["uci"]) == nxt), "?")
        moves.append(played)
    vd = exs[-1].visit_distribution
    j = max(range(len(vd)), key=lambda k: vd[k])
    moves.append(exs[-1].legal_moves[j]["uci"])
    return exs[0].fen, moves


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Watch a greedy (argmax) self-play game from a FEN, OR replay a "
                    "given movelist (--moves) — e.g. paste a selfplay PGN line.")
    ap.add_argument("fen", nargs="?", default=DEFAULT_START_FEN,
                    help=f"Chessckers start FEN (default: the simplified training start "
                         f"{DEFAULT_START_FEN!r})")
    ap.add_argument("--moves", default="",
                    help="replay this movelist instead of having the net play. Accepts a raw "
                         "selfplay PGN line (a leading 'PGN:' and a trailing result/'{OL...}' "
                         "comment are ignored). Each token is a Chessckers UCI, e.g. "
                         "'a2a4 e6f7 c3:h5~e2->e2 d6c5[1] ...'.")
    ap.add_argument("--chunk", default="",
                    help="replay a RECORDED ccz1 self-play game (e.g. "
                         "../lczero-server/games/run1/training.N.gz). Recovers the start FEN and the "
                         "ACTUAL played moves from the chunk (not the visit-argmax) and renders it. "
                         "Overrides the positional FEN and --moves.")
    ap.add_argument("--no-eval", action="store_true",
                    help="replay only: skip loading the net / printing per-ply WDL eval (faster).")
    ap.add_argument("--weights", default="", help="checkpoint .pt (default: latest weights/run/{weights.pt,iter-async-*.pt}, then base_wdl_v*.pt)")
    ap.add_argument("--latest", action="store_true",
                    help="use the live published fleet net (lczero-server/trainer/run1/weights.pt), "
                         "freshest first. Combine with no FEN to watch the latest net from the "
                         "engine's current start position: `watch_game.py --latest --device mps`.")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--explore", type=float, default=0.30,
                    help="root Dirichlet exploration-noise fraction (default 0.30 = 30 pct); the "
                         "played move stays argmax of visits. 0 = pure greedy/deterministic.")
    ap.add_argument("--seed", type=int, default=-1,
                    help="rng seed (default: random each run, so games vary)")
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--device", default="cpu", help="cpu|mps|cuda (default cpu)")
    ap.add_argument("--delay", type=float, default=0.0, help="extra pause between plies, seconds")
    ap.add_argument("--step", action="store_true",
                    help="step through plies interactively: pause after each and wait for a key "
                         "(space/Enter = next, q = quit). Implies --clear.")
    ap.add_argument("--clear", action="store_true",
                    help="clear the screen before each ply so the game renders in place instead "
                         "of scrolling.")
    args = ap.parse_args()
    if args.step:
        args.clear = True  # stepping in place is the point; clear unless replaying to a pipe

    import torch
    from chessckers_engine.checkpoints import load_scorer
    from chessckers_engine.mcts_puct import run_mcts
    from chessckers_engine.render_board import render_board
    from chessckers_engine.selfplay_az import _outcome_from_state
    from chessckers_engine.variant_py import PyVariantClient

    if args.chunk:  # recorded game: derive start FEN + real move line, then take the --moves path
        args.fen, _chunk_moves = _moves_from_chunk(args.chunk)
        args.moves = " ".join(_chunk_moves)
        print(f"chunk: {args.chunk}  ({len(_chunk_moves)} plies)")

    replay = bool(args.moves)
    need_model = not (replay and args.no_eval)  # replay --no-eval needs no net at all

    model = None
    weights = None
    if need_model:
        last_err: Exception | None = None
        for cand in _resolve_weights(args.weights, args.latest):  # freshest first; weights.pt may be mid-write
            try:
                # load_scorer reads the .arch.json sidecar and builds the EXACT arch (V1/V2/V3), so a
                # tf=7 V3 checkpoint loads into a V3 model instead of silently dropping most weights
                # into a V1 shell (the strict=False footgun this helper exists to close).
                model = load_scorer(cand).to(args.device).eval()
                weights = cand
                break
            except Exception as e:  # noqa: BLE001 — try the next durable candidate
                last_err = e
        if weights is None:
            raise SystemExit(f"could not load any candidate checkpoint; last error: {last_err}")
        assert model is not None  # set iff weights was set; the SystemExit above guards the None case

    seed = args.seed if args.seed >= 0 else int.from_bytes(os.urandom(4), "big")
    # run_mcts draws root Dirichlet noise from the GLOBAL torch RNG, so seed it
    # here: this makes --seed actually reproduce a game (and --seed -1 vary it).
    torch.manual_seed(seed)
    print(f"weights: {weights}\nsims: {args.sims} | device: {args.device} | "
          f"explore (root noise): {args.explore:.0%} | move pick: argmax | seed: {seed}\n")

    os.environ["CHESSCKERS_START_FEN"] = args.fen  # new_game() reads this
    client = PyVariantClient()
    state = client.new_game()
    alpha = 0.3 if args.explore > 0 else None  # AlphaZero-chess default concentration

    def show_board(ply: int, uci: str | None, fen: str) -> None:
        head = f"ply {ply}: {uci}" if uci else "start"
        print(f"\n=== {head} ===")
        print(render_board(fen))

    if args.clear:
        _clear_screen()
    show_board(0, None, state["fen"])
    if not _advance(args):
        return 0

    if replay:
        return _replay(args, client, state, model, show_board, _outcome_from_state)

    assert model is not None  # non-replay path always loads the net (need_model)
    ply = 0
    while not state.get("status") and ply < args.max_plies:
        if not (state.get("legalMoves") or []):
            break
        if args.clear:
            _clear_screen()
        result = run_mcts(
            state, client, model,
            n_sims=args.sims, c_puct=1.5,
            dirichlet_alpha=alpha,      # root exploration noise...
            dirichlet_eps=args.explore,  # ...at --explore fraction (30%)
        )
        _print_net_eval(model, state["fen"], state["turn"], args.device)  # raw net WDL + moves-left
        _print_analysis(state["turn"], result, args.sims)  # top-3 lines for this position
        chosen = result.chosen
        if chosen is None:
            break
        state = client.make_move(state["fen"], chosen["uci"])
        ply += 1
        show_board(ply, chosen["uci"], state["fen"])
        if not _advance(args):
            break

    status = state.get("status")
    if status:
        outcome = _outcome_from_state(state)
        print(f"\n######## {outcome.upper()} WINS ({status}) in {ply} plies ########"
              if outcome != "draw" else f"\n######## DRAW ({status}) in {ply} plies ########")
    else:
        print(f"\n######## UNFINISHED — stopped at {ply} plies (max-plies={args.max_plies}) ########")
    return 0


if __name__ == "__main__":
    sys.exit(main())
