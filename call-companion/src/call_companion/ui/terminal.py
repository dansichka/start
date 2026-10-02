"""Plain console view of the event stream; works in any terminal and over SSH."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape

from ..bus import Event, EventBus

LABELS = {"answer": "ОТВЕТ", "fact": "ФАКТ", "reminder": "НАПОМИНАНИЕ", "ask": "СПРОСИТЬ", "warn": "ВНИМАНИЕ"}
STYLES = {"answer": "bold green", "fact": "bold cyan", "reminder": "bold magenta", "ask": "bold blue", "warn": "bold red"}


class TerminalView:
    def __init__(self, console: Console | None = None, transcript: bool = True) -> None:
        self.console = console or Console()
        self.transcript = transcript

    def render(self, event: Event) -> None:
        data, say = event.data, self.console.print
        if event.type == "brief":
            say(f"[bold]Досье:[/] {escape(data['title'])} · материалов: {data['items']} · режим: {data['mode']}")
            for loop in data.get("open_loops", []):
                say(f"  [yellow]↺ хвост:[/] {escape(loop)}")
        elif event.type == "utterance" and self.transcript:
            who = "я" if data["channel"] == "me" else data["speaker"]
            say(f"[dim]{escape(who)}:[/] {escape(data['text'])}")
        elif event.type == "suggestion" and data.get("final"):
            kind = data["kind"]
            say(f"[{STYLES[kind]}]▌ {LABELS[kind]}[/] {escape(data['text'])}")
            sources = ", ".join(f"{s['source']}: {s['title'] or s['id']}" for s in data.get("sources", []))
            if sources:
                say(f"  [dim]источники: {escape(sources)}[/]")
        elif event.type == "action_candidate":
            due = f" ({escape(data['due'])})" if data.get("due") else ""
            say(f"[cyan]☐ {escape(data['owner'] or '?')}:[/] {escape(data['text'])}{due}")
        elif event.type == "approval":
            say(f"[magenta]подтверждение[/] {escape(data['action']['title'])} — {data['status']}")
        elif event.type == "warning":
            say(f"[yellow]! {escape(data['message'])}[/]")
        elif event.type == "error":
            say(f"[red]✗ {escape(data['message'])}[/]")
        elif event.type == "clear":
            say("[dim]— экран очищен —[/]")
        elif event.type == "status" and data.get("state") == "ended":
            say("[bold]Звонок завершён.[/]")

    async def run(self, bus: EventBus, after: int | None = 0) -> None:
        async for event in bus.subscribe(after):
            if event is not None:
                self.render(event)
