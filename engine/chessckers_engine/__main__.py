import logging
import os
import sys

import httpx

from chessckers_engine.http_server import make_server
from chessckers_engine.server_client import ServerClient


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("chessckers_engine")

    api_url = os.environ.get("API_URL", "http://localhost:8080")
    host = os.environ.get("ENGINE_HOST", "127.0.0.1")
    port = int(os.environ.get("ENGINE_PORT", "8082"))

    client = ServerClient(base_url=api_url)
    try:
        client.new_game()
    except httpx.ConnectError:
        log.error("cannot reach API at %s (start the server first)", api_url)
        return 1

    server = make_server(host, port, client)
    log.info("random-move opponent listening on http://%s:%d/move", host, port)
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
