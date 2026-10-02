"""FastAPI transport for the kernel contract; runs on the kernel host next to DataSource."""

from __future__ import annotations

import hmac
import json
from typing import Any

from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .kernel_api import DataSource, UnknownOp, dispatch

MAX_BODY_BYTES = 1_000_000


def create_app(source: DataSource, token: str) -> FastAPI:
    """Build the API; refuses to start without a token."""
    if not token:
        raise ValueError("token обязателен: без него сервис нельзя поднимать")
    expected = token.encode()
    app = FastAPI(title="Kernel companion API", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip().encode()
        if not hmac.compare_digest(supplied, expected):
            return JSONResponse({"detail": "bad token"}, status_code=401)
        declared = request.headers.get("content-length") or "0"
        if not declared.isdigit():
            return JSONResponse({"detail": "bad content-length"}, status_code=400)
        if int(declared) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "body too large"}, status_code=413)
        return await call_next(request)

    def call(op: str, payload: dict[str, Any]) -> JSONResponse | dict[str, Any]:
        try:
            return dispatch(source, op, payload)
        except ValidationError as exc:
            return JSONResponse({"detail": json.loads(exc.json())}, status_code=422)
        except UnknownOp as exc:
            return JSONResponse({"detail": f"unknown op {exc}"}, status_code=404)

    @app.get("/v1/health")
    def health():
        return call("health", {})

    @app.post("/v1/search")
    def search(payload: dict[str, Any] = Body(...)):
        return call("search", payload)

    @app.post("/v1/brief")
    def brief(payload: dict[str, Any] = Body(...)):
        return call("brief", payload)

    @app.get("/v1/meetings/upcoming")
    def upcoming(within_min: int = 30):
        return call("upcoming", {"within_min": within_min})

    @app.post("/v1/notes")
    def note(payload: dict[str, Any] = Body(...)):
        return call("note", payload)

    @app.post("/v1/actions")
    def action(payload: dict[str, Any] = Body(...)):
        return call("action", payload)

    return app


def serve(source: DataSource, token: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(create_app(source, token), host=host, port=port, log_level="warning")
