"""Watch the network's opinion drift across checkpoints.

For a fixed bank of positions, this prints each board once (with the rim, via
render_board) and then one row per checkpoint showing the value head and the
top policy moves. Frozen rows across iterations = stale; value sharpening and
policy concentrating = the net is learning.

    python probe_drift.py weights/iter-az-001.pt weights/iter-az-002.pt ...
    python probe_drift.py --dir weights/ln_v2          # all *.pt in a dir, sorted

Architecture (c_filters / d_hidden / n_blocks) is auto-detected from each
checkpoint's state_dict, so no manual flags. Checkpoints from an incompatible
older topology (no residual tower, or a non-240 move encoding) are reported
and skipped rather than crashing the run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from chessckers_engine.encoding import MOVE_D, encode_move, encode_position
from chessckers_engine.render_board import render_board
from chessckers_engine.variant_py import PyVariantClient
from chessckers_engine.variant_py.state import STARTING_FEN, parse_fen

TOP_K = 5


def detect_arch(sd: dict) -> dict:
    """Infer ChesskersScorer constructor args from a state_dict. Raises
    ValueError if the checkpoint is from an incompatible older topology."""
    if "position_trunk.0.weight" not in sd or "move_encoder.0.weight" not in sd:
        raise ValueError("missing expected keys — not a ChesskersScorer checkpoint")
    c_filters = sd["position_trunk.0.weight"].shape[0]
    d_hidden, d_move = sd["move_encoder.0.weight"].shape
    n_blocks = len({
        k.split(".")[1] for k in sd
        if k.startswith("position_trunk.") and k.endswith(".conv1.weight")
    })
    if n_blocks == 0:
        raise ValueError("no residual tower — pre-residual checkpoint, incompatible")
    if d_move != MOVE_D:
        raise ValueError(f"move encoding is {d_move}-dim, current is {MOVE_D} — incompatible")
    return {"c_filters": c_filters, "d_hidden": d_hidden, "n_blocks": n_blocks}


def load_model(path: Path, device: str):
    """Return (model, arch) or (None, reason) if the checkpoint can't be loaded."""
    from chessckers_engine.model import ChesskersScorer
    sd = torch.load(path, map_location="cpu", weights_only=True)
    try:
        arch = detect_arch(sd)
    except ValueError as e:
        return None, str(e)
    model = ChesskersScorer(**arch).to(device).eval()
    model.load_state_dict(sd)
    return model, arch


def eval_position(model, fen: str, moves: list[dict], device: str):
    """Return (value, [(move, prior), ...] sorted by prior desc)."""
    pos_t = encode_position(fen).unsqueeze(0).to(device)
    move_t = torch.stack([encode_move(m) for m in moves]).to(device)
    with torch.no_grad():
        logits, value = model.policy_and_value(pos_t, move_t)
        priors = torch.softmax(logits, dim=0).cpu().tolist()
    rows = sorted(zip(moves, priors), key=lambda r: r[1], reverse=True)
    return float(value.cpu()), rows


def _tags(m: dict) -> str:
    t = []
    if m.get("isCapture"):
        t.append("x")
    if m.get("waypoints"):
        t.append(f"ch{len(m['waypoints'])}")
    return ("+" + ",".join(t)) if t else ""


def build_bank() -> list[tuple[str, str]]:
    """Starting position + two early derived positions (deterministic)."""
    client = PyVariantClient()
    bank = [("starting position (white to move)", STARTING_FEN)]
    s = client.make_move(STARTING_FEN, "e2e4")
    bank.append(("after 1.e4 (black to move)", s["fen"]))
    moves = sorted(s["legalMoves"], key=lambda m: (m.get("isCapture", False), m.get("uci", "")))
    if moves:
        s2 = client.make_move(s["fen"], moves[0]["uci"])
        bank.append((f"after 1.e4 {moves[0]['uci']} (white to move)", s2["fen"]))
    return bank


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="*", type=Path, help="checkpoint .pt files, in order")
    p.add_argument("--dir", type=Path, default=None, help="use all *.pt in this dir, sorted by name")
    p.add_argument("--device", default="cpu")
    p.add_argument("--top-k", type=int, default=TOP_K)
    args = p.parse_args()

    paths = sorted(args.dir.glob("*.pt")) if args.dir else list(args.checkpoints)
    if not paths:
        print("no checkpoints given (pass paths or --dir)", file=sys.stderr)
        return 2

    client = PyVariantClient()
    models: list[tuple[str, object]] = []
    for path in paths:
        model, info = load_model(path, args.device)
        if model is None:
            print(f"skip {path.name}: {info}")
            continue
        models.append((path.name, model))
    if not models:
        print("no loadable checkpoints — nothing to compare", file=sys.stderr)
        return 1
    print(f"comparing {len(models)} checkpoint(s): {', '.join(n for n, _ in models)}\n")

    for label, fen in build_bank():
        state = parse_fen(fen)
        status, _winner, moves = client.status_and_legal(state)
        print(f"=== {label} ===")
        print(render_board(fen))
        if status is not None or not moves:
            print("(terminal position)\n")
            continue
        moves = list(moves)
        name_w = max(len(n) for n, _ in models)
        for name, model in models:
            value, rows = eval_position(model, fen, moves, args.device)
            top = "  ".join(f"{m.get('uci','?')}{_tags(m)} {pr*100:.0f}%"
                            for m, pr in rows[: args.top_k])
            print(f"  {name:<{name_w}}  value={value:+.3f}  | {top}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
