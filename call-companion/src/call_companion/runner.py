"""Run one call end to end: HUD, console, a driver that feeds the engine, then summary and approvals."""

from __future__ import annotations

import asyncio
import signal
import sys
import webbrowser
from dataclasses import dataclass
from typing import Awaitable, Callable

from rich.console import Console
from rich.markup import escape

from .app import CallOutcome, Runtime, finish_call
from .live.engine import LiveEngine
from .models import Meeting, MeetingMode
from .ui.server import create_app, hud_url, serve
from .ui.terminal import TerminalView

YES = {"y", "yes", "д", "да"}
TICK_S = 0.25


@dataclass
class SessionOptions:
    meeting: Meeting | None = None
    mode: MeetingMode | None = None
    ui: bool = True
    terminal: bool = True
    compact: bool = False
    open_browser: bool = False
    summarize: bool = True
    auto_approve: bool = False
    ticker: bool = True
    token: str | None = None
    on_ready: Callable[[str], None] | None = None


def install_signals(stop: asyncio.Event) -> Callable[[], None]:
    """Ctrl+C and SIGTERM end the call gracefully; returns a function that restores the defaults."""
    loop = asyncio.get_running_loop()
    installed = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
            installed.append(sig)
        except (NotImplementedError, RuntimeError, ValueError):  # Windows, or not the main thread
            pass

    def restore() -> None:
        for sig in installed:
            loop.remove_signal_handler(sig)

    return restore


async def _ticker(engine: LiveEngine) -> None:
    while True:
        await asyncio.sleep(TICK_S)
        engine.tick()


def print_outcome(console: Console, outcome: CallOutcome) -> None:
    summary = outcome.summary
    if summary is None:
        console.print("[yellow]Итоги не получились: расшифровка сохранена, смотрите ошибки выше.[/]")
    else:
        console.print(f"\n[bold]Итоги:[/] {escape(summary.title)}\n{escape(summary.tldr)}")
        for heading, rows in (("Решения", summary.decisions), ("Открытые вопросы", summary.open_questions)):
            if rows:
                console.print(f"[bold]{heading}:[/]")
                for row in rows:
                    console.print(f"  • {escape(row)}")
        if summary.action_items:
            console.print("[bold]Обещания и задачи:[/]")
            for item in summary.action_items:
                due = f" (до: {escape(item.due)})" if item.due else ""
                console.print(f"  ☐ {escape(item.owner or '?')}: {escape(item.text)}{due}")
    pending = [t for t in outcome.tickets if t.status == "pending"]
    if pending:
        console.print(f"[magenta]Ждут вашего решения: {len(pending)}[/] (записи и задачи в kernel)")
    draft = outcome.session_dir / "followup.md"
    if draft.exists():
        console.print(f"Черновик follow-up (не отправлен): {draft}")


async def _settle_approvals(rt: Runtime, opts: SessionOptions, outcome: CallOutcome, stop: asyncio.Event, console: Console) -> None:
    pending = [t for t in outcome.tickets if t.status == "pending"]
    if not pending:
        return
    if opts.auto_approve:
        for ticket in pending:
            await rt.engine.decide(ticket.id, True, "cli")
        console.print(f"[green]Подтверждено флагом --yes: {len(pending)}[/]")
    elif opts.ui:
        console.print("Подтвердите действия в HUD. Ctrl+C — выйти.")
        stop.clear()
        await stop.wait()
    elif sys.stdin.isatty():
        answer = await asyncio.to_thread(input, f"Записать {len(pending)} в kernel? [y/N] ")
        if answer.strip().lower() in YES:
            for ticket in pending:
                await rt.engine.decide(ticket.id, True, "cli")


async def run_session(
    rt: Runtime,
    opts: SessionOptions,
    driver: Callable[[], Awaitable[None]],
    stop: asyncio.Event | None = None,
    console: Console | None = None,
) -> CallOutcome:
    stop = stop or asyncio.Event()
    console = console or Console()
    restore_signals = install_signals(stop)
    view = asyncio.create_task(TerminalView(console).run(rt.bus, after=0)) if opts.terminal else None
    server = None
    try:
        if opts.ui:
            app = create_app(rt.engine, rt.bus, rt.settings, token=opts.token, on_end=stop.set)
            server = await serve(app, rt.settings)
            url = hud_url(rt.settings, app.state.token, compact=opts.compact)
            console.print(f"HUD: {url}")
            if opts.on_ready:
                opts.on_ready(url)
            if opts.open_browser:
                webbrowser.open(url)
        await rt.engine.start(opts.meeting, opts.mode)

        driving = asyncio.create_task(driver())
        helpers = [asyncio.create_task(stop.wait())]
        if opts.ticker:
            helpers.append(asyncio.create_task(_ticker(rt.engine)))
        await asyncio.wait({driving, helpers[0]}, return_when=asyncio.FIRST_COMPLETED)
        for task in (driving, *helpers):
            task.cancel()
        results = await asyncio.gather(driving, *helpers, return_exceptions=True)
        failure = results[0]
        if isinstance(failure, Exception):
            rt.bus.publish("error", message=f"{type(failure).__name__}: {failure}")

        outcome = await finish_call(rt, summarize=opts.summarize)
        print_outcome(console, outcome)
        await _settle_approvals(rt, opts, outcome, stop, console)
        return outcome
    finally:
        restore_signals()
        if server is not None:
            server.should_exit = True
            await app.state.server_task
        if view is not None:
            await asyncio.sleep(0.05)
            view.cancel()
            await asyncio.gather(view, return_exceptions=True)
        await rt.kernel.aclose()
