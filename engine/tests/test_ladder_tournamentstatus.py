"""parse_tournamentstatus: the selfplay-harness contract with the fork's output.

The ladder's gate-harness mode (ladder.py --engine, default since 07-17) scores
pairs entirely from the engine's `tournamentstatus` stream — a parse regression
would silently zero the audit, so the line shapes are pinned here. Samples are
verbatim from /workspace/diag27_sp128.log and the 07-17 probe.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from ladder import parse_tournamentstatus  # noqa: E402


def test_running_line():
    st = parse_tournamentstatus(
        "tournamentstatus P1: +42 -117 =0 Win: 26.42% Elo: -177.97 LOS:  0.00% "
        "P1-W: +37 -42 =0 P1-B: +5 -75 =0 npm 137.543844 nodes 1497990 moves 10891")
    assert st == {"w": 42, "l": 117, "d": 0, "ww": 37, "wl": 42, "wd": 0,
                  "bw": 5, "bl": 75, "bd": 0, "final": False}


def test_final_line():
    st = parse_tournamentstatus(
        "tournamentstatus final P1: +99 -60 =0 Win: 62.26% Elo: 86.99 LOS: 99.90% "
        "P1-W: +76 -4 =0 P1-B: +23 -56 =0 npm 138.257633 nodes 1507976 moves 10907")
    assert st["final"] is True
    assert (st["w"], st["l"], st["d"]) == (99, 60, 0)
    # color physics: White wins = P1-as-W wins + P1-as-B losses
    assert st["ww"] + st["bl"] == 76 + 56


def test_elo_field_omitted_at_lopsided_score():
    # At 100%/0% the engine omits `Elo:` entirely — the parser must not need it.
    st = parse_tournamentstatus(
        "tournamentstatus final P1: +2 -0 =0 Win: 100.00% LOS: 92.14% "
        "P1-W: +1 -0 =0 P1-B: +1 -0 =0 npm 17.255172 nodes 2502 moves 145")
    assert st == {"w": 2, "l": 0, "d": 0, "ww": 1, "wl": 0, "wd": 0,
                  "bw": 1, "bl": 0, "bd": 0, "final": True}


def test_draws_counted():
    st = parse_tournamentstatus(
        "tournamentstatus P1: +10 -8 =4 Win: 54.55% Elo: 31.65 LOS: 82.00% "
        "P1-W: +7 -2 =2 P1-B: +3 -6 =2 npm 100.0 nodes 1000 moves 100")
    assert st["d"] == 4 and st["wd"] + st["bd"] == 4


def test_non_status_lines_ignored():
    assert parse_tournamentstatus("gameready gameid 4 play_start_ply 0 player1 "
                                  "black result whitewon moves b1c3 g1f3") is None
    assert parse_tournamentstatus("Chessckers backend: CUDA GPU trunk enabled") is None
