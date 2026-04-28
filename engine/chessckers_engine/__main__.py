import logging
import os
import sys
from pathlib import Path

import httpx

from chessckers_engine.checkpoints import latest_checkpoint
from chessckers_engine.http_server import make_server
from chessckers_engine.runtime import build_pickers
from chessckers_engine.server_client import ServerClient

DEFAULT_GAMES_PATH = Path(__file__).resolve().parent.parent / "games" / "games.jsonl"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("chessckers_engine")

    api_url = os.environ.get("API_URL", "http://localhost:8080")
    host = os.environ.get("ENGINE_HOST", "127.0.0.1")
    port = int(os.environ.get("ENGINE_PORT", "8082"))
    default_picker = os.environ.get("ENGINE_DEFAULT_PICKER", "random")
    mcts_sims = int(os.environ.get("ENGINE_MCTS_SIMS", "100"))

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

    pickers = build_pickers(client, model_path, log, mcts_sims=mcts_sims)
    if default_picker not in pickers:
        log.error("ENGINE_DEFAULT_PICKER=%r is not in available pickers %s", default_picker, sorted(pickers))
        client.close()
        return 2

    games_path_str = os.environ.get("ENGINE_GAMES_PATH", str(DEFAULT_GAMES_PATH))
    games_path = Path(games_path_str) if games_path_str else None
    server = make_server(host, port, client, pickers, default_picker=default_picker, games_path=games_path)
    log.info("listening on http://%s:%d/move; pickers=%s; default=%s; games=%s",
             host, port, sorted(pickers), default_picker, games_path)
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
