import io

from rich.console import Console

from call_companion.bus import EventBus
from call_companion.ui.terminal import TerminalView


def render(*events: tuple[str, dict], transcript: bool = True) -> str:
    out = io.StringIO()
    view = TerminalView(Console(file=out, width=110, color_system=None, force_terminal=False), transcript=transcript)
    bus = EventBus()
    for type_, data in events:
        view.render(bus.publish(type_, **data))
    return out.getvalue()


def test_final_whispers_are_printed_with_their_sources_and_partials_are_not():
    text = render(
        ("suggestion", {"id": "s", "kind": "answer", "text": "Бюд", "final": False}),
        (
            "suggestion",
            {
                "id": "s", "kind": "warn", "text": "Не 18 500, а 18 400 EUR", "final": True,
                "sources": [{"source": "note", "title": "Бюджет Orion", "id": "note:55"}],
            },
        ),
    )
    assert "Бюд\n" not in text
    assert "ВНИМАНИЕ Не 18 500, а 18 400 EUR" in text and "источники: note: Бюджет Orion" in text


def test_call_text_cannot_inject_console_markup():
    text = render(("utterance", {"id": "u", "channel": "them", "speaker": "[bold red]Evil[/]", "text": "[red]hack[/red] \\"}))
    assert "[bold red]Evil[/]: [red]hack[/red]" in text


def test_other_events_have_readable_lines():
    text = render(
        ("brief", {"title": "Статус Orion", "items": 7, "mode": "client", "open_loops": ["Обещал план"]}),
        ("action_candidate", {"id": "a", "text": "Я пришлю план", "owner": "я", "due": "до пятницы"}),
        ("approval", {"id": "t", "status": "pending", "action": {"title": "Записать факт"}}),
        ("warning", {"message": "kernel не ответил"}),
        ("error", {"message": "boom"}),
        ("clear", {}),
        ("status", {"state": "ended"}),
    )
    for fragment in ("Досье: Статус Orion · материалов: 7", "хвост: Обещал план", "☐ я: Я пришлю план (до пятницы)",
                     "Записать факт — pending", "! kernel не ответил", "✗ boom", "экран очищен", "Звонок завершён"):
        assert fragment in text


def test_transcript_lines_can_be_switched_off():
    event = ("utterance", {"id": "u", "channel": "me", "speaker": "me", "text": "привет"})
    assert "я: привет" in render(event)
    assert render(event, transcript=False) == ""
