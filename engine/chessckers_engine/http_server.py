import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from chessckers_engine.server_client import ServerClient

log = logging.getLogger("chessckers_engine.http")

GameState = dict[str, Any]
LegalMove = dict[str, Any]
Picker = Callable[[GameState], LegalMove | None]

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


class EngineHandler(BaseHTTPRequestHandler):
    client: ServerClient
    picker: Picker

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/move":
            self._send_json(404, {"error": f"unknown path {self.path}"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        fen = body.get("fen")
        if not isinstance(fen, str) or not fen:
            self._send_json(400, {"error": "missing 'fen'"})
            return
        try:
            state = self.client.new_game(fen)
        except Exception as e:
            self._send_json(502, {"error": f"upstream API failed: {e}"})
            return
        chosen = self.picker(state)
        self._send_json(200, {"uci": chosen["uci"] if chosen else None})


def make_server(host: str, port: int, client: ServerClient, picker: Picker) -> ThreadingHTTPServer:
    handler_cls = type(
        "BoundEngineHandler", (EngineHandler,), {"client": client, "picker": staticmethod(picker)}
    )
    ThreadingHTTPServer.allow_reuse_address = True
    return ThreadingHTTPServer((host, port), handler_cls)
