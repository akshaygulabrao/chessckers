import sys

import httpx

from chessckers_engine import ServerClient


def main() -> int:
    try:
        with ServerClient() as c:
            state = c.new_game()
    except httpx.ConnectError:
        print("engine: cannot reach API at http://localhost:8080 (start the server first)")
        return 1
    print("engine: connected to server")
    print(f"engine: starting FEN = {state['fen']}")
    print(f"engine: legal moves at start = {len(state['legalMoves'])}")
    print(f"engine: turn = {state['turn']}")
    print("engine: placeholder run complete (self-play loop lands in milestone 3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
