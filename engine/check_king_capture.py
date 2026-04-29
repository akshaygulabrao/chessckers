"""Diagnostic: can Black capture the White king via a diagonal jump?

Constructs a minimal position:
   - White king alone on c4
   - Black single stone (height 1) on b5
   - Black to move

Expected per spec: Black's b5 stack should be able to jump *over* c4 onto
d3, capturing the king. This is a diagonal-capture chain of length 1.
After execution, status should be 'mate' (Black wins by king capture).

If the king-capture move is missing from legalMoves, or if the server
rejects the move when applied, that confirms the alleged bug.
"""
from __future__ import annotations

import json

from chessckers_engine.server_client import ServerClient

# Black stone (lowercase p in standard FEN) at b5; White king at c4.
# Bracket overlay declares b5 as a height-1 unmoved stone.
FEN = "8/8/8/1p6/2K5/8/8/8[b5:s] b - - 0 1"


def main() -> int:
    client = ServerClient()
    try:
        # /new accepts arbitrary FEN; returns full state including legalMoves.
        state = client.new_game(FEN)
    except Exception as e:  # noqa: BLE001
        print(f"server unreachable: {e}")
        return 1

    print(f"FEN: {FEN}")
    print(f"turn: {state['turn']}")
    print(f"check: {state['check']}")
    print(f"status: {state['status']}")
    print(f"legalMoves total: {len(state['legalMoves'])}")
    from_b5 = [m for m in state["legalMoves"] if m.get("from") == "b5"]
    print(f"\nlegal moves from b5 ({len(from_b5)}):")
    for m in from_b5:
        print(f"  {json.dumps(m)}")

    target = next((m for m in from_b5 if m.get("to") == "d3"), None)
    if target is None:
        print("\n*** b5→d3 (king capture) is NOT in the legal-moves list. ***")
        print("    This is consistent with the user's bug report.")
        client.close()
        return 0

    print(f"\nb5→d3 is in legalMoves. Trying to execute uci={target['uci']!r}…")
    try:
        next_state = client.make_move(state["fen"], target["uci"])
    except Exception as e:  # noqa: BLE001
        print(f"*** make_move REJECTED: {e}")
        client.close()
        return 0

    print("post-move FEN:", next_state["fen"])
    print("post-move turn:", next_state["turn"])
    print("post-move status:", next_state["status"])
    print("post-move winner:", next_state.get("winner"))
    if next_state.get("status") == "mate":
        print("\n[OK] Black wins by king capture, as expected.")
    else:
        print("\n*** Move executed but status is not 'mate'.")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
