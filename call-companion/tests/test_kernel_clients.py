import asyncio
import json
import os
import sys
import time

import httpx
import pytest

from call_companion.kernel.base import KernelAuthError, KernelError, KernelUnavailable
from call_companion.kernel.cli import CliKernel
from call_companion.kernel.composite import CompositeKernel
from call_companion.kernel.http import HttpKernel
from call_companion.kernel.memory import InMemoryKernel
from call_companion.kernel.snapshot import SnapshotKernel
from call_companion.kernel_api import MemoryDataSource, write_pack
from call_companion.models import Note, SearchFilters

from .conftest import item


async def test_snapshot_loads_pack_and_skips_broken_lines(tmp_path, items, entities, meeting):
    pack = tmp_path / "pack.jsonl"
    write_pack(pack, items, entities, [meeting])
    with pack.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
        handle.write('{"type":"mystery"}\n')
    kernel = SnapshotKernel(pack)
    assert (await kernel.search("бюджет этапа", k=2))[0].id == "note:55"
    assert kernel.skipped == 2
    assert (await kernel.health()).items == len(items)


async def test_snapshot_reloads_when_file_changes(tmp_path, items):
    pack = tmp_path / "pack.jsonl"
    write_pack(pack, items[2:3])
    kernel = SnapshotKernel(pack)
    assert await kernel.search("бюджет этапа") == []
    write_pack(pack, items)
    stat = pack.stat()
    os.utime(pack, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert (await kernel.search("бюджет этапа"))[0].id == "note:55"


async def test_snapshot_missing_file_is_unavailable(tmp_path):
    with pytest.raises(KernelUnavailable):
        await SnapshotKernel(tmp_path / "nope.jsonl").search("x")


async def test_snapshot_notes_go_to_outbox_not_into_the_pack(tmp_path, items):
    pack = tmp_path / "pack.jsonl"
    write_pack(pack, items)
    before = pack.read_text(encoding="utf-8")
    kernel = SnapshotKernel(pack)
    note_id = await kernel.save_note(Note(text="Договорились о демо 8 октября", tags=["orion"]))
    assert pack.read_text(encoding="utf-8") == before
    row = json.loads(kernel.outbox_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["id"] == note_id and row["type"] == "note" and row["untrusted"] is True


def http_kernel(handler, **kwargs) -> HttpKernel:
    return HttpKernel("http://kernel", "tok", transport=httpx.MockTransport(handler), **kwargs)


def ok_search(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"items": [{"id": "a", "source": "note", "text": "привет", "score": 3.0}]})


async def test_http_sends_token_and_body():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return ok_search(request)

    kernel = http_kernel(handler)
    found = await kernel.search("бюджет", k=4, filters=SearchFilters(max_sensitivity="private"))
    assert found[0].id == "a"
    assert seen["auth"] == "Bearer tok"
    assert seen["body"]["k"] == 4 and seen["body"]["filters"]["max_sensitivity"] == "private"


async def test_http_caches_identical_searches():
    calls = []

    def handler(request):
        calls.append(1)
        return ok_search(request)

    kernel = http_kernel(handler)
    await kernel.search("бюджет")
    await kernel.search("бюджет")
    await kernel.search("сроки")
    assert len(calls) == 2


async def test_http_timeout_and_network_errors_are_unavailable():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(KernelUnavailable):
        await http_kernel(handler).search("x")


async def test_http_circuit_breaker_opens_and_recovers():
    clock = [100.0]
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503)

    kernel = http_kernel(handler, breaker_threshold=3, breaker_reset_s=20, clock=lambda: clock[0], cache_ttl_s=0)
    for _ in range(3):
        with pytest.raises(KernelUnavailable):
            await kernel.search("x")
    assert len(calls) == 3
    with pytest.raises(KernelUnavailable, match="цепь разомкнута"):
        await kernel.search("x")
    assert len(calls) == 3
    clock[0] += 21
    with pytest.raises(KernelUnavailable):
        await kernel.search("x")
    assert len(calls) == 4


async def test_http_auth_errors_are_distinct_and_do_not_trip_the_breaker():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401)

    kernel = http_kernel(handler, breaker_threshold=2, cache_ttl_s=0)
    for _ in range(5):
        with pytest.raises(KernelAuthError):
            await kernel.search("x")
    assert len(calls) == 5


async def test_http_client_errors_and_garbage():
    with pytest.raises(KernelError) as bad_request:
        await http_kernel(lambda r: httpx.Response(422, text="nope")).search("x")
    assert not isinstance(bad_request.value, KernelUnavailable)
    with pytest.raises(KernelError, match="не JSON"):
        await http_kernel(lambda r: httpx.Response(200, text="<html>")).search("x")


async def test_http_skips_invalid_items_but_keeps_valid_ones():
    def handler(request):
        return httpx.Response(200, json={"items": [{"source": "no-id"}, {"id": "ok", "source": "note"}]})

    assert [i.id for i in await http_kernel(handler).search("x")] == ["ok"]


CLI_SCRIPT = """
import json, sys, time
op = sys.argv[1]
request = json.loads(sys.stdin.buffer.read().decode("utf-8") or "{}")
mode = __MODE__
if mode == "slow":
    time.sleep(5)
if mode == "crash":
    sys.stderr.write("boom"); sys.exit(3)
if mode == "garbage":
    print("not json"); sys.exit(0)
if op == "health":
    print(json.dumps({"ok": True, "version": "fake", "items": 1}))
elif op == "search":
    row = {"id": "x1", "source": "note", "text": "эхо: " + request["query"]}
    sys.stdout.buffer.write(json.dumps({"items": [row]}, ensure_ascii=False).encode("utf-8"))
elif op == "note":
    print(json.dumps({"id": "n1"}))
"""


def cli_kernel(tmp_path, mode="ok", timeout_s=3.0) -> CliKernel:
    script = tmp_path / f"kernel_{mode}.py"
    script.write_text(CLI_SCRIPT.replace("__MODE__", repr(mode)), encoding="utf-8")
    return CliKernel([sys.executable, str(script)], timeout_s=timeout_s)


async def test_cli_round_trip_keeps_utf8(tmp_path):
    kernel = cli_kernel(tmp_path)
    assert (await kernel.health()).version == "fake"
    found = await kernel.search("бюджет проекта «Орион»")
    assert found[0].text == "эхо: бюджет проекта «Орион»"
    assert await kernel.save_note(Note(text="x")) == "n1"


async def test_cli_timeout_crash_garbage_and_missing_binary(tmp_path):
    with pytest.raises(KernelUnavailable, match="не уложился"):
        await cli_kernel(tmp_path, "slow", timeout_s=0.3).search("x")
    with pytest.raises(KernelError, match="boom"):
        await cli_kernel(tmp_path, "crash").search("x")
    with pytest.raises(KernelError, match="не JSON"):
        await cli_kernel(tmp_path, "garbage").search("x")
    with pytest.raises(KernelUnavailable):
        await CliKernel(["definitely-not-a-binary-xyz"]).search("x")


async def test_composite_merges_dedupes_and_normalizes(items):
    kernel = CompositeKernel([InMemoryKernel(MemoryDataSource(items[:3])), InMemoryKernel(MemoryDataSource(items[1:5]))])
    results = await kernel.search("бюджет этапа Orion", k=5)
    assert len({r.id for r in results}) == len(results)
    assert results[0].score == 1.0 and all(0 < r.score <= 1.0 for r in results)


async def test_composite_slow_backend_is_cut_by_the_budget(items):
    kernel = CompositeKernel(
        [InMemoryKernel(MemoryDataSource(items)), InMemoryKernel(MemoryDataSource(items), latency_s=2.0)],
        search_budget_s=0.15,
    )
    started = time.monotonic()
    results = await kernel.search("бюджет", k=3)
    assert results and time.monotonic() - started < 0.8


async def test_composite_survives_partial_failure_and_reports_total_failure(items):
    healthy = InMemoryKernel(MemoryDataSource(items))
    broken = InMemoryKernel(fail=True)
    assert await CompositeKernel([broken, healthy]).search("бюджет")
    with pytest.raises(KernelUnavailable):
        await CompositeKernel([InMemoryKernel(fail=True), InMemoryKernel(fail=True)]).search("x")


async def test_composite_notes_prefer_live_kernel_then_fall_back_to_outbox(tmp_path, items):
    pack = tmp_path / "pack.jsonl"
    write_pack(pack, items)
    snapshot = SnapshotKernel(pack)
    live = InMemoryKernel()
    await CompositeKernel([snapshot, live]).save_note(Note(text="жив"))
    assert len(live.source.notes) == 1 and not snapshot.outbox_path.exists()

    await CompositeKernel([snapshot, InMemoryKernel(fail=True)]).save_note(Note(text="офлайн"))
    assert "офлайн" in snapshot.outbox_path.read_text(encoding="utf-8")


async def test_composite_actions_need_a_live_backend(tmp_path, items):
    from call_companion.models import ProposedAction

    pack = tmp_path / "pack.jsonl"
    write_pack(pack, items)
    action = ProposedAction(id="a1", capability="kernel.task", title="t")
    with pytest.raises(KernelUnavailable):
        await CompositeKernel([SnapshotKernel(pack)]).submit_action(action)
    live = InMemoryKernel()
    assert await CompositeKernel([SnapshotKernel(pack), live]).submit_action(action) == "a1"


async def test_composite_brief_merges_and_health_aggregates(items, entities, meeting):
    local = InMemoryKernel(MemoryDataSource(items[:2], entities))
    remote = InMemoryKernel(MemoryDataSource(items[2:5]))
    kernel = CompositeKernel([local, remote])
    brief = await kernel.brief(meeting)
    assert {"card:101", "gh:pr-214"} <= {i.id for i in brief.items}
    assert (await kernel.health()).items == 5


async def test_composite_requires_backends():
    with pytest.raises(ValueError):
        CompositeKernel([])


async def test_in_memory_kernel_failure_injection_and_latency():
    kernel = InMemoryKernel(MemoryDataSource([item("a", "привет мир")]), latency_s=0.05)
    started = time.monotonic()
    assert await kernel.search("привет")
    assert time.monotonic() - started >= 0.05
    kernel.fail = True
    with pytest.raises(KernelUnavailable):
        await kernel.search("привет")
    assert kernel.calls == ["search", "search"]
