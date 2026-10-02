import asyncio
import json
import socket
from pathlib import Path

import httpx
import pytest

from call_companion.app import finish_call
from call_companion.config import UiCfg
from call_companion.ui import server as server_module
from call_companion.ui.server import create_app, hud_url, serve

from .support import demo_runtime, play_demo, running

TOKEN = "tok-123"
JSON_HEADERS = {"X-Companion": "1", "Content-Type": "application/json"}


@pytest.fixture
async def hud(tmp_path):
    runtime, clock = await demo_runtime(tmp_path)
    return runtime, clock, create_app(runtime.engine, runtime.bus, runtime.settings, token=TOKEN)


def client(app, **kwargs) -> httpx.AsyncClient:
    kwargs.setdefault("cookies", {"callc_token": TOKEN})
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost", **kwargs)


async def test_token_is_required_and_swapped_for_a_cookie(hud):
    _, _, app = hud
    async with client(app, cookies={}) as http:
        assert (await http.get("/api/state")).status_code == 401
        assert (await http.get("/?token=wrong")).status_code == 401
        redirect = await http.get("/?token=" + TOKEN + "&compact=1")
        assert redirect.status_code == 303 and redirect.headers["location"] == "/?compact=1"
        cookie = redirect.headers["set-cookie"].lower()
        assert "callc_token=" + TOKEN in cookie and "httponly" in cookie and "samesite=strict" in cookie
        assert (await http.get("/api/state")).status_code == 200
    async with client(app, cookies={}) as http:
        assert (await http.get("/api/state", headers={"X-Token": TOKEN})).status_code == 200


async def test_host_header_is_checked_against_dns_rebinding(hud):
    runtime, _, app = hud
    async with client(app) as http:
        assert (await http.get("/api/state", headers={"Host": "evil.example"})).status_code == 400
        assert (await http.get("/api/state", headers={"Host": "localhost:7777"})).status_code == 200
        assert (await http.get("/api/state", headers={"Host": "100.101.102.103:7777"})).status_code == 200
        assert (await http.get("/api/state", headers={"Host": "[::1]:7777"})).status_code == 200
        assert (await http.get("/api/state", headers={"Host": "mac.local:7777"})).status_code == 400
    named = create_app(
        runtime.engine, runtime.bus, runtime.settings.model_copy(update={"ui": UiCfg(allowed_hosts=["mac.local"])}), token=TOKEN
    )
    async with client(named) as http:
        assert (await http.get("/api/state", headers={"Host": "mac.local:7777"})).status_code == 200


async def test_changes_need_the_custom_header_and_a_same_site_origin(hud):
    runtime, _, app = hud
    async with client(app) as http:
        body = json.dumps({"muted": True})
        assert (await http.post("/api/mute", content=body)).status_code == 403
        forged = {**JSON_HEADERS, "Origin": "http://evil.example"}
        assert (await http.post("/api/mute", content=body, headers=forged)).status_code == 403
        assert not runtime.engine.gate.muted
        ok = {**JSON_HEADERS, "Origin": "http://localhost:7777"}
        assert (await http.post("/api/mute", content=body, headers=ok)).status_code == 200
        assert runtime.engine.gate.muted


async def test_ask_mode_panic_and_validation(hud):
    runtime, _, app = hud
    async with client(app) as http:
        post = lambda path, body: http.post(path, content=json.dumps(body), headers=JSON_HEADERS)
        assert (await post("/api/ask", {"text": "какой бюджет этапа 1?"})).status_code == 200
        await runtime.engine.settle()
        assert runtime.engine.suggestions[-1].trigger == "manual"
        assert (await post("/api/mode", {"mode": "internal"})).status_code == 200
        assert runtime.engine.mode == "internal"
        assert (await post("/api/mode", {"mode": "root"})).status_code == 422
        assert (await post("/api/ask", {"text": "x" * 501})).status_code == 422
        assert (await post("/api/ask", {"text": "ok", "extra": 1})).status_code == 422
        assert (await http.post("/api/ask", content="{not json", headers=JSON_HEADERS)).status_code == 422
        assert (await post("/api/panic", {})).status_code == 200
        assert runtime.engine.gate.muted and any(e.type == "clear" for e in runtime.bus.history())


async def test_end_callback(tmp_path):
    runtime, _ = await demo_runtime(tmp_path)
    ended = []
    app = create_app(runtime.engine, runtime.bus, runtime.settings, token=TOKEN, on_end=lambda: ended.append(1))
    async with client(app) as http:
        assert (await http.post("/api/end", content="{}", headers=JSON_HEADERS)).status_code == 200
    assert ended == [1]


async def test_approvals_through_the_hud(hud):
    runtime, clock, app = hud
    await play_demo(runtime, clock)
    outcome = await finish_call(runtime)
    note = next(t for t in outcome.tickets if t.action.capability == "kernel.note")
    async with client(app) as http:
        post = lambda ticket, approve: http.post(f"/api/approvals/{ticket}", content=json.dumps({"approve": approve}), headers=JSON_HEADERS)
        done = await post(note.id, True)
        assert done.status_code == 200 and done.json()["status"] == "executed"
        assert (await post(note.id, True)).status_code == 409
        assert (await post("tk_missing", True)).status_code == 404


async def test_state_snapshot_survives_a_reload(hud):
    runtime, clock, app = hud
    await play_demo(runtime, clock)
    await finish_call(runtime)
    async with client(app) as http:
        state = (await http.get("/api/state")).json()
    assert state["state"] == "ended" and state["summary"]["title"] == "Статус Orion"
    assert state["last_event_id"] == runtime.bus.last_id
    assert all(s["sources"] for s in state["suggestions"] if s["evidence"])
    assert len(state["approvals"]) == 4 and state["actions"]


def test_the_page_never_writes_untrusted_text_as_html():
    page = (Path(server_module.__file__).with_name("static") / "hud.html").read_text(encoding="utf-8")
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert banned not in page, banned


async def test_the_page_is_served_without_caching(hud):
    _, _, app = hud
    async with client(app) as http:
        page = await http.get("/")
    assert page.status_code == 200 and page.headers["cache-control"] == "no-store" and "Компаньон" in page.text


async def test_sse_replays_history_then_follows_and_pings(hud, monkeypatch):
    runtime, _, app = hud
    monkeypatch.setattr(server_module, "HEARTBEAT_S", 0.05)
    async with running(app) as base:
        async with httpx.AsyncClient(base_url=base, cookies={"callc_token": TOKEN}, timeout=5) as http:
            async with http.stream("GET", "/events?after=0") as response:
                assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")

                async def later():
                    await asyncio.sleep(0.2)
                    runtime.bus.publish("warning", message="sse-marker-live")

                task = asyncio.create_task(later())
                seen: list[str] = []
                async with asyncio.timeout(5):
                    async for line in response.aiter_lines():
                        seen.append(line)
                        if line.startswith("data: ") and "sse-marker-live" in line:
                            break
                await task
    text = "\n".join(seen)
    assert "event: brief" in text and "event: status" in text and ": ping" in text
    assert json.loads(next(l for l in seen if "sse-marker-live" in l)[6:]) == {"message": "sse-marker-live"}


async def test_sse_resumes_after_last_event_id(hud):
    runtime, _, app = hud
    last = runtime.bus.last_id
    async with running(app) as base:
        async with httpx.AsyncClient(base_url=base, cookies={"callc_token": TOKEN}, timeout=5) as http:
            async with http.stream("GET", "/events", headers={"Last-Event-ID": str(last)}) as response:
                async def later():
                    await asyncio.sleep(0.1)
                    runtime.bus.publish("warning", message="sse-marker-resume")

                task = asyncio.create_task(later())
                frames: list[str] = []
                async with asyncio.timeout(5):
                    async for line in response.aiter_lines():
                        frames.append(line)
                        if line.startswith("data: "):
                            break
                await task
    assert frames[-1] == 'data: {"message": "sse-marker-resume"}'
    assert "event: brief" not in "\n".join(frames)


async def test_serve_starts_and_stops_the_server(hud):
    runtime, _, app = hud
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    settings = runtime.settings.model_copy(update={"ui": UiCfg(port=port)})
    server = await serve(app, settings)
    try:
        async with httpx.AsyncClient(cookies={"callc_token": TOKEN}) as http:
            assert (await http.get(f"http://127.0.0.1:{port}/api/state")).status_code == 200
    finally:
        server.should_exit = True
        await app.state.server_task
    assert hud_url(settings, "abc") == f"http://127.0.0.1:{port}/?token=abc"
    assert hud_url(settings, "abc", compact=True).endswith("&compact=1")
    wide = settings.model_copy(update={"ui": UiCfg(host="0.0.0.0", port=port)})
    assert hud_url(wide, "abc").startswith("http://localhost:")
