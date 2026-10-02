import asyncio
import builtins
import io
import os
import signal
import socket
import sys
import threading
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from rich.console import Console

from call_companion.app import build_runtime
from call_companion.config import PrivacyCfg, Settings, UiCfg, UserCfg
from call_companion.kernel.memory import InMemoryKernel
from call_companion.llm.fake import FakeLLM
from call_companion.models import Utterance
from call_companion.runner import SessionOptions, install_signals, run_session

from .test_engine import BrokenLLM

JSON = {"X-Companion": "1", "Content-Type": "application/json"}


def make_runtime(tmp_path, port: int | None = None, llm=None):
    ui = UiCfg(port=port) if port else UiCfg()
    settings = Settings(user=UserCfg(name="Даня"), ui=ui, privacy=PrivacyCfg(data_dir=str(tmp_path)))
    kernel = InMemoryKernel()
    return build_runtime(settings, llm=llm or FakeLLM(), kernel=kernel, ephemeral=True), kernel


def says(rt, text: str = "Я пришлю план до пятницы") -> None:
    rt.engine.feed(Utterance(id=f"u{len(rt.engine.transcript.finals)}", channel="me", speaker="me", text=text, t_start=0, t_end=2))


def quiet() -> Console:
    return Console(file=io.StringIO(), width=100)


def options(**kwargs) -> SessionOptions:
    return SessionOptions(**{"ui": False, "terminal": False, "ticker": False, **kwargs})


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


async def test_the_session_wraps_up_when_the_driver_finishes(tmp_path):
    rt, _ = make_runtime(tmp_path)

    async def driver():
        says(rt)

    outcome = await run_session(rt, options(), driver, console=quiet())
    assert len(outcome.record.utterances) == 1 and len(outcome.record.action_items) == 1
    assert outcome.summary is not None
    assert sorted(t.action.capability for t in outcome.tickets if t.status == "pending") == ["kernel.note", "kernel.task"]


async def test_the_stop_event_ends_a_driver_that_never_returns(tmp_path):
    rt, _ = make_runtime(tmp_path)
    stop, cancelled = asyncio.Event(), []

    async def driver():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    asyncio.get_running_loop().call_later(0.1, stop.set)
    outcome = await asyncio.wait_for(run_session(rt, options(), driver, stop=stop, console=quiet()), 5)
    assert cancelled == [True] and outcome.record.ended_at is not None


async def test_a_crashing_driver_still_wraps_up_the_call(tmp_path):
    rt, _ = make_runtime(tmp_path)

    async def driver():
        says(rt)
        raise RuntimeError("микрофон отвалился")

    outcome = await run_session(rt, options(), driver, console=quiet())
    assert outcome.summary is not None and len(outcome.record.utterances) == 1
    assert any(e.type == "error" and "микрофон отвалился" in e.data["message"] for e in rt.bus.history())


async def test_a_failing_model_still_leaves_the_transcript_and_files(tmp_path):
    rt, _ = make_runtime(tmp_path, llm=BrokenLLM())

    async def driver():
        says(rt)

    outcome = await run_session(rt, options(), driver, console=quiet())
    assert outcome.summary is None and (outcome.session_dir / "meta.json").exists()


async def test_yes_flag_approves_everything_pending(tmp_path):
    rt, kernel = make_runtime(tmp_path)

    async def driver():
        says(rt)

    await run_session(rt, options(auto_approve=True), driver, console=quiet())
    assert len(kernel.source.notes) == 1 and len(kernel.source.actions) == 1


@pytest.mark.parametrize("answer, written", [("y", 1), ("да", 1), ("n", 0), ("", 0)])
async def test_the_terminal_prompt_decides_when_there_is_no_hud(tmp_path, monkeypatch, answer, written):
    rt, kernel = make_runtime(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda prompt="": answer)

    async def driver():
        says(rt)

    await run_session(rt, options(), driver, console=quiet())
    assert len(kernel.source.notes) == written


async def test_the_hud_is_served_during_the_call_and_approvals_are_decided_there(tmp_path):
    port = free_port()
    rt, kernel = make_runtime(tmp_path, port=port)
    url_box: asyncio.Queue = asyncio.Queue()

    async def driver():
        says(rt)
        await asyncio.sleep(0.3)

    async def human():
        url = await asyncio.wait_for(url_box.get(), 5)
        token = parse_qs(urlsplit(url).query)["token"][0]
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", cookies={"callc_token": token}) as http:
            state = {}
            for _ in range(100):
                state = (await http.get("/api/state")).json()
                if state["state"] == "ended" and state["approvals"]:
                    break
                await asyncio.sleep(0.05)
            note = next(t for t in state["approvals"] if t["action"]["capability"] == "kernel.note")
            done = await http.post(f"/api/approvals/{note['id']}", content='{"approve": true}', headers=JSON)
            assert done.json()["status"] == "executed"
            assert (await http.post("/api/end", content="{}", headers=JSON)).status_code == 200

    opts = options(ui=True, on_ready=url_box.put_nowait)
    outcome, _ = await asyncio.wait_for(asyncio.gather(run_session(rt, opts, driver, console=quiet()), human()), 20)
    assert len(kernel.source.notes) == 1 and kernel.source.actions == []
    assert outcome.summary is not None


async def test_ending_the_call_from_the_hud_stops_a_running_session(tmp_path):
    port = free_port()
    rt, _ = make_runtime(tmp_path, port=port)
    url_box: asyncio.Queue = asyncio.Queue()

    async def driver():
        await asyncio.sleep(3600)

    async def human():
        token = parse_qs(urlsplit(await asyncio.wait_for(url_box.get(), 5)).query)["token"][0]
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", cookies={"callc_token": token}) as http:
            await http.post("/api/end", content="{}", headers=JSON)

    opts = options(ui=True, on_ready=url_box.put_nowait, summarize=False)
    await asyncio.wait_for(asyncio.gather(run_session(rt, opts, driver, console=quiet()), human()), 20)


async def test_sigint_ends_the_call_gracefully():
    stop = asyncio.Event()
    restore = install_signals(stop)
    try:
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.wait_for(stop.wait(), 2)
    finally:
        restore()
    assert stop.is_set()


def test_signal_handlers_are_skipped_quietly_outside_the_main_thread():
    result = []

    def worker():
        async def main():
            restore = install_signals(asyncio.Event())
            restore()
            result.append("ok")

        asyncio.run(main())

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(5)
    assert result == ["ok"]
