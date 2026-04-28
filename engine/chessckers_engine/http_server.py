import json
import logging
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
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
    pickers: dict[str, Picker]
    default_picker_name: str
    games_path: Path | None  # if set, /save-game appends finished games here

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
        if self.path == "/save-game":
            self._handle_save_game()
            return
        if self.path != "/move":
            self._send_json(404, {"error": f"unknown path {self.path}"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        fen = body.get("fen")
        if not isinstance(fen, str) or not fen:
            self._send_json(400, {"error": "missing 'fen'"})
            return

        picker_name = body.get("picker") or self.default_picker_name
        picker = self.pickers.get(picker_name)
        if picker is None:
            self._send_json(
                400,
                {"error": f"unknown picker {picker_name!r}; known: {sorted(self.pickers)}"},
            )
            return

        try:
            state = self.client.new_game(fen)
        except Exception as e:
            self._send_json(502, {"error": f"upstream API failed: {e}"})
            return
        chosen = picker(state)
        self._send_json(200, {"uci": chosen["uci"] if chosen else None})

    def _handle_save_game(self) -> None:
        if self.games_path is None:
            self._send_json(503, {"error": "save-game disabled (no games path configured)"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        # The body should contain at minimum: history (list of {fen, uci}), final_status, outcome.
        if "history" not in body or "outcome" not in body:
            self._send_json(400, {"error": "missing 'history' or 'outcome'"})
            return
        # Stamp it server-side so the file's well-formed regardless of client.
        body.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
        try:
            self.games_path.parent.mkdir(parents=True, exist_ok=True)
            with self.games_path.open("a") as f:
                f.write(json.dumps(body))
                f.write("\n")
        except Exception as e:  # noqa: BLE001
            self._send_json(500, {"error": f"write failed: {e}"})
            return
        self._send_json(200, {"saved": True, "path": str(self.games_path)})


def make_server(
    host: str,
    port: int,
    client: ServerClient,
    pickers: dict[str, Picker],
    default_picker: str = "random",
    games_path: Path | None = None,
) -> ThreadingHTTPServer:
    if default_picker not in pickers:
        raise ValueError(f"default picker {default_picker!r} not in pickers {sorted(pickers)}")
    handler_cls = type(
        "BoundEngineHandler",
        (EngineHandler,),
        {
            "client": client,
            "pickers": pickers,
            "default_picker_name": default_picker,
            "games_path": games_path,
        },
    )
    ThreadingHTTPServer.allow_reuse_address = True
    return ThreadingHTTPServer((host, port), handler_cls)
