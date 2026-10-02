"""The real HttpKernel against the real create_app, and CliKernel against run_cli."""

import io
import json
import sys
import textwrap
from pathlib import Path

import httpx
import pytest

from call_companion.kernel.base import KernelAuthError
from call_companion.kernel.cli import CliKernel
from call_companion.kernel.http import HttpKernel
from call_companion.kernel_api import MAX_K, MemoryDataSource, run_cli
from call_companion.kernel_server import create_app
from call_companion.models import Note, ProposedAction, SearchFilters

from .conftest import item

TOKEN = "s3cret-token"


def client_for(app, token: str | None = TOKEN) -> HttpKernel:
    return HttpKernel("http://kernel", token, transport=httpx.ASGITransport(app=app), cache_ttl_s=0)


@pytest.fixture
def source(items, entities, meeting) -> MemoryDataSource:
    return MemoryDataSource(items, entities, [meeting.model_copy(update={"id": "soon"})])


async def test_every_operation_round_trips(source, meeting):
    kernel = client_for(create_app(source, TOKEN))
    assert (await kernel.health()).ok
    found = await kernel.search("какой бюджет первого этапа", k=3)
    assert found[0].id == "note:55" and found[0].score > 0
    brief = await kernel.brief(meeting)
    assert {"card:101", "card:130"} <= {i.id for i in brief.items}
    assert brief.open_loops and brief.talking_points
    note_id = await kernel.save_note(Note(text="Согласовали демо на 8 октября", tags=["orion"]))
    assert note_id.startswith("note_") and source.notes[0].untrusted
    action_id = await kernel.submit_action(ProposedAction(id="act1", capability="kernel.task", title="Отправить план"))
    assert action_id == "act1" and source.actions[0].title == "Отправить план"


async def test_bad_or_missing_token_is_rejected(source):
    app = create_app(source, TOKEN)
    with pytest.raises(KernelAuthError):
        await client_for(app, "wrong").health()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://kernel") as raw:
        assert (await raw.get("/v1/health")).status_code == 401
        assert (await raw.post("/v1/search", json={"query": "x"})).status_code == 401


def test_service_refuses_to_start_without_a_token(source):
    with pytest.raises(ValueError):
        create_app(source, "")


async def test_server_enforces_sensitivity_even_if_the_data_source_leaks(items):
    class Leaky(MemoryDataSource):
        def search(self, query, k, filters):
            return list(self.items)

    kernel = client_for(create_app(Leaky(items), TOKEN))
    leaked = {i.id for i in await kernel.search("что угодно", filters=SearchFilters(max_sensitivity="work"))}
    assert "tg:private1" not in leaked and "stripe:9" not in leaked
    allowed = {i.id for i in await kernel.search("что угодно", filters=SearchFilters(max_sensitivity="private"))}
    assert "tg:private1" in allowed and "stripe:9" not in allowed


async def test_result_count_is_clamped(items):
    many = [item(f"n{i}", "общий текст про Orion") for i in range(60)]
    kernel = client_for(create_app(MemoryDataSource(many), TOKEN))
    assert len(await kernel.search("Orion", k=500)) == MAX_K


async def test_invalid_payload_unknown_path_and_hidden_docs(source):
    app = create_app(source, TOKEN)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://kernel", headers=headers) as raw:
        assert (await raw.post("/v1/search", json={"nope": 1})).status_code == 422
        assert (await raw.get("/v1/nothing")).status_code == 404
        assert (await raw.get("/docs")).status_code == 404
        assert (await raw.get("/openapi.json")).status_code == 404


async def test_oversized_body_is_refused(source):
    app = create_app(source, TOKEN)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://kernel", headers=headers) as raw:
        response = await raw.post("/v1/notes", content=b"x" * 1_100_000, headers={"content-type": "application/json"})
        assert response.status_code == 413


async def test_a_garbage_content_length_is_a_client_error_not_a_crash(source):
    app = create_app(source, TOKEN)
    headers = {"Authorization": f"Bearer {TOKEN}", "content-length": "abc"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://kernel") as raw:
        assert (await raw.get("/v1/health", headers=headers)).status_code == 400


CLI_WRAPPER = """
import sys
sys.path.insert(0, {src!r})
from call_companion.kernel_api import MemoryDataSource, run_cli
from call_companion.models import ContextItem
data = MemoryDataSource([
    ContextItem(id="c1", source="note", title="Бюджет Orion", text="Бюджет этапа 1 — 18 400 EUR", entities=["Orion"]),
    ContextItem(id="c2", source="stripe", text="секрет", sensitivity="secret"),
])
raise SystemExit(run_cli(data))
"""


async def test_cli_wrapper_matches_the_contract(tmp_path):
    script = tmp_path / "kernel_companion.py"
    script.write_text(CLI_WRAPPER.format(src=str(Path(__file__).parents[1] / "src")), encoding="utf-8")
    kernel = CliKernel([sys.executable, str(script)], timeout_s=10)
    found = await kernel.search("бюджет этапа", filters=SearchFilters(max_sensitivity="private"))
    assert [i.id for i in found] == ["c1"]
    assert "18 400" in found[0].text
    assert (await kernel.health()).items == 2
    assert (await kernel.save_note(Note(text="x"))).startswith("note_")


def test_run_cli_usage_and_errors(monkeypatch, capsys):
    source = MemoryDataSource()
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert run_cli(source, ["unknown-op"]) == 2
    assert run_cli(source, []) == 2
    monkeypatch.setattr(sys, "stdin", io.StringIO("{broken"))
    assert run_cli(source, ["search"]) == 1
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"nope": 1})))
    assert run_cli(source, ["search"]) == 1
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert run_cli(source, ["health"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
