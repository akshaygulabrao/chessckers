import logging
import os
import sys

import httpx

from chessckers_engine.http_server import GameState, LegalMove, Picker, make_server
from chessckers_engine.random_player import pick_random
from chessckers_engine.server_client import ServerClient


def _build_random_picker() -> Picker:
    def picker(state: GameState) -> LegalMove | None:
        return pick_random(state.get("legalMoves") or [])

    return picker


def _build_nn_picker(model_path: str | None, log: logging.Logger) -> Picker:
    import torch

    from chessckers_engine.model import ChesskersScorer
    from chessckers_engine.nn_player import pick_nn

    model = ChesskersScorer()
    if model_path:
        log.info("loading model weights from %s", model_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    else:
        log.info("no ENGINE_MODEL set; using random-init weights (plays random-ish)")
    model.eval()

    def picker(state: GameState) -> LegalMove | None:
        return pick_nn(state, model)

    return picker


def _select_picker(player: str, model_path: str | None, log: logging.Logger) -> tuple[Picker, str]:
    if player == "random":
        return _build_random_picker(), "random-move"
    if player == "nn":
        return _build_nn_picker(model_path, log), "neural-net"
    raise ValueError(f"ENGINE_PLAYER must be 'random' or 'nn'; got {player!r}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("chessckers_engine")

    api_url = os.environ.get("API_URL", "http://localhost:8080")
    host = os.environ.get("ENGINE_HOST", "127.0.0.1")
    port = int(os.environ.get("ENGINE_PORT", "8082"))
    player = os.environ.get("ENGINE_PLAYER", "random")
    model_path = os.environ.get("ENGINE_MODEL") or None

    client = ServerClient(base_url=api_url)
    try:
        client.new_game()
    except httpx.ConnectError:
        log.error("cannot reach API at %s (start the server first)", api_url)
        return 1

    try:
        picker, label = _select_picker(player, model_path, log)
    except ValueError as e:
        log.error(str(e))
        client.close()
        return 2

    server = make_server(host, port, client, picker)
    log.info("%s opponent listening on http://%s:%d/move", label, host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        server.server_close()
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
