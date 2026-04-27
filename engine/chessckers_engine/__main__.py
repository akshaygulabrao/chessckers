import logging
import os
import sys

import httpx

from chessckers_engine.checkpoints import latest_checkpoint
from chessckers_engine.http_server import GameState, LegalMove, Picker, make_server
from chessckers_engine.material_player import pick_material
from chessckers_engine.random_player import pick_random
from chessckers_engine.server_client import ServerClient


def _build_pickers(client: ServerClient, model_path: str | None, log: logging.Logger) -> dict[str, Picker]:
    pickers: dict[str, Picker] = {}

    def random_picker(state: GameState) -> LegalMove | None:
        return pick_random(state.get("legalMoves") or [])

    pickers["random"] = random_picker

    def material_picker(state: GameState) -> LegalMove | None:
        return pick_material(state, client)

    pickers["material"] = material_picker

    # NN picker (lazy: only import torch if requested)
    try:
        import torch

        from chessckers_engine.model import ChesskersScorer
        from chessckers_engine.nn_player import pick_nn

        model = ChesskersScorer()
        if model_path:
            log.info("loading NN weights from %s", model_path)
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
        else:
            log.info("no ENGINE_MODEL set; NN picker uses random-init weights")
        model.eval()

        def nn_picker(state: GameState) -> LegalMove | None:
            return pick_nn(state, model)

        pickers["nn"] = nn_picker
    except Exception as e:  # noqa: BLE001
        log.warning("NN picker unavailable (%s: %s); 'random' and 'material' still work", type(e).__name__, e)

    return pickers


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("chessckers_engine")

    api_url = os.environ.get("API_URL", "http://localhost:8080")
    host = os.environ.get("ENGINE_HOST", "127.0.0.1")
    port = int(os.environ.get("ENGINE_PORT", "8082"))
    default_picker = os.environ.get("ENGINE_DEFAULT_PICKER", "random")

    # ENGINE_MODEL takes precedence; if unset, auto-pick the latest .pt under weights/.
    model_path = os.environ.get("ENGINE_MODEL")
    if not model_path:
        latest = latest_checkpoint()
        if latest is not None:
            model_path = str(latest)
            log.info("auto-selected latest checkpoint: %s", model_path)
    model_path = model_path or None

    client = ServerClient(base_url=api_url)
    try:
        client.new_game()
    except httpx.ConnectError:
        log.error("cannot reach API at %s (start the server first)", api_url)
        return 1

    pickers = _build_pickers(client, model_path, log)
    if default_picker not in pickers:
        log.error("ENGINE_DEFAULT_PICKER=%r is not in available pickers %s", default_picker, sorted(pickers))
        client.close()
        return 2

    server = make_server(host, port, client, pickers, default_picker=default_picker)
    log.info("listening on http://%s:%d/move; pickers=%s; default=%s",
             host, port, sorted(pickers), default_picker)
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
