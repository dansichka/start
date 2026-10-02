"""HUD web server: serves the page, streams events over SSE and accepts the few commands a human may give.

Secure by default: every request needs the token (cookie, header or one-time ?token=), the Host header must be
a loopback name, an IP literal or an allowed name (blocks DNS rebinding), and state-changing requests must carry
a custom header and a same-site Origin, so a web page you happen to browse cannot drive the companion.
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import secrets
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..bus import EventBus
from ..config import Settings
from ..live.engine import LiveEngine
from ..models import MeetingMode

STATIC = Path(__file__).with_name("static")
COOKIE = "callc_token"
HEARTBEAT_S = 15.0


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AskBody(_Body):
    text: str = Field(default="", max_length=500)


class MuteBody(_Body):
    muted: bool


class ModeBody(_Body):
    mode: MeetingMode


class ApproveBody(_Body):
    approve: bool


def _hostname(value: str) -> str:
    return (urlsplit(f"//{value}").hostname or "").lower()


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def sse_frame(event) -> str:
    return f"id: {event.id}\nevent: {event.type}\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"


def create_app(
    engine: LiveEngine,
    bus: EventBus,
    settings: Settings,
    *,
    token: str | None = None,
    on_end: Callable[[], None] | None = None,
) -> FastAPI:
    token = token or settings.ui.token or secrets.token_urlsafe(18)
    allowed_names = {"localhost", *(h.lower() for h in settings.ui.allowed_hosts)}
    page = (STATIC / "hud.html").read_text(encoding="utf-8")
    app = FastAPI(title="Call companion HUD", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.token = token

    def host_allowed(value: str) -> bool:
        host = _hostname(value)
        return host in allowed_names or _is_ip(host)

    def token_ok(candidate: str | None) -> bool:
        return bool(candidate) and hmac.compare_digest(candidate.encode(), token.encode())

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if not host_allowed(request.headers.get("host", "")):
            return JSONResponse({"detail": "bad host"}, status_code=400)
        offered = request.query_params.get("token")
        if request.method == "GET" and token_ok(offered):
            target = request.url.path + (f"?{_without_token(request)}" if _without_token(request) else "")
            response = RedirectResponse(target, status_code=303)
            response.set_cookie(COOKIE, token, httponly=True, samesite="strict", path="/")
            return response
        if not token_ok(request.cookies.get(COOKIE)) and not token_ok(request.headers.get("x-token")):
            return JSONResponse({"detail": "token required"}, status_code=401)
        if request.method != "GET":
            origin = request.headers.get("origin")
            if request.headers.get("x-companion") != "1" or (origin and not host_allowed(origin.split("//", 1)[-1])):
                return JSONResponse({"detail": "forbidden"}, status_code=403)
        return await call_next(request)

    def _without_token(request: Request) -> str:
        return "&".join(f"{k}={v}" for k, v in request.query_params.multi_items() if k != "token")

    async def parsed(request: Request, model: type[_Body]):
        try:
            return model.model_validate(await request.json()), None
        except (ValidationError, ValueError) as exc:
            return None, JSONResponse({"detail": str(exc)[:200]}, status_code=422)

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(page, headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    async def state() -> dict:
        return {**engine.state(), "last_event_id": bus.last_id}

    @app.get("/events")
    async def events(request: Request, after: int | None = None) -> StreamingResponse:
        header = request.headers.get("last-event-id")
        resume = after if after is not None else (int(header) if header and header.isdigit() else None)

        async def stream():
            subscription = bus.subscribe(resume, heartbeat_s=HEARTBEAT_S)
            try:
                yield ": connected\n\n"
                async for event in subscription:
                    yield ": ping\n\n" if event is None else sse_frame(event)
            finally:
                await subscription.aclose()

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
        )

    @app.post("/api/ask")
    async def ask(request: Request):
        body, error = await parsed(request, AskBody)
        if error:
            return error
        engine.ask(body.text)
        return {"ok": True}

    @app.post("/api/mute")
    async def mute(request: Request):
        body, error = await parsed(request, MuteBody)
        if error:
            return error
        engine.mute(body.muted)
        return {"ok": True}

    @app.post("/api/mode")
    async def mode(request: Request):
        body, error = await parsed(request, ModeBody)
        if error:
            return error
        await engine.set_mode(body.mode)
        return {"ok": True}

    @app.post("/api/panic")
    async def panic():
        engine.panic()
        return {"ok": True}

    @app.post("/api/approvals/{ticket_id}")
    async def approve(ticket_id: str, request: Request):
        body, error = await parsed(request, ApproveBody)
        if error:
            return error
        try:
            ticket = await engine.decide(ticket_id, body.approve, "hud")
        except KeyError:
            return JSONResponse({"detail": "нет такого подтверждения"}, status_code=404)
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=409)
        return ticket.to_dict()

    @app.post("/api/end")
    async def end():
        if on_end:
            on_end()
        return {"ok": True}

    return app


def hud_url(settings: Settings, token: str, *, compact: bool = False) -> str:
    host = settings.ui.host if settings.ui.host not in ("0.0.0.0", "::") else "localhost"
    return f"http://{host}:{settings.ui.port}/?token={token}" + ("&compact=1" if compact else "")


async def serve(app: FastAPI, settings: Settings) -> uvicorn.Server:
    """Start the server in the running loop; the caller keeps the task and sets server.should_exit to stop it."""
    config = uvicorn.Config(
        app, host=settings.ui.host, port=settings.ui.port, log_level="warning", timeout_graceful_shutdown=2
    )
    server = uvicorn.Server(config)
    app.state.server_task = asyncio.create_task(server.serve())
    while not server.started:
        if app.state.server_task.done():
            app.state.server_task.result()
        await asyncio.sleep(0.02)
    return server
