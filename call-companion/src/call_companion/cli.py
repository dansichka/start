"""`callc`: the command line of the call companion."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import doctor
from .app import Runtime, build_runtime
from .asr.engines import build_asr
from .asr.pipeline import AsrPipeline
from .asr.replay import load_transcript, replay
from .clock import VirtualClock
from .config import ConfigError, Settings, load_settings
from .kernel import KernelError, build_kernel
from .kernel.snapshot import SnapshotKernel
from .kernel_api import MemoryDataSource, read_pack
from .meeting.detector import pick_meeting, running_call_apps
from .models import ActionItem, CallRecord, Meeting, Participant, Utterance
from .policy import AuditLog, PolicyEngine
from .post.store import SessionStore
from .post.summary import summarize_call
from .runner import SessionOptions, run_session
from .util import new_id

MODES = ("internal", "client", "interview", "public")


def has_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or (Path.home() / ".config" / "anthropic").exists())


def load(args: argparse.Namespace) -> Settings:
    settings = load_settings(args.config)
    name = getattr(args, "user", None)
    if name:
        user = settings.user.model_copy(update={"name": name, "aliases": list(dict.fromkeys([name, *settings.user.aliases]))})
        settings = settings.model_copy(update={"user": user})
    return settings


def build_meeting(args: argparse.Namespace) -> Meeting | None:
    if getattr(args, "meeting", None):
        return Meeting.model_validate_json(Path(args.meeting).read_text(encoding="utf-8"))
    if getattr(args, "title", None) or getattr(args, "attendee", None):
        people = []
        for entry in args.attendee or []:
            name, _, org = entry.partition(":")
            people.append(Participant(name=name.strip(), org=org.strip() or None))
        return Meeting(id=new_id("adhoc"), title=args.title or "Созвон", agenda=args.agenda or "", attendees=people)
    return None


async def choose_meeting(rt: Runtime, args: argparse.Namespace, console: Console) -> Meeting | None:
    explicit = build_meeting(args)
    if explicit or not (args.meeting_id or args.auto):
        return explicit
    announced = False
    while True:
        try:
            meetings = await rt.kernel.upcoming(60)
        except KernelError as exc:
            console.print(f"[yellow]календарь недоступен: {escape(str(exc))}[/]")
            meetings = []
        chosen = next((m for m in meetings if m.id == args.meeting_id), None) if args.meeting_id else pick_meeting(meetings)
        if chosen or not args.auto:
            if chosen:
                console.print(f"Встреча: [bold]{escape(chosen.title)}[/]; приложения для звонков: {', '.join(running_call_apps()) or 'не видно'}")
            return chosen
        if not announced:
            console.print("Жду встречу из календаря… (Ctrl+C — отмена)")
            announced = True
        await asyncio.sleep(20)


def asr_prompt(engine) -> str | None:
    """Names from the brief help Whisper spell proper nouns it has never heard."""
    names = [p.name for p in (engine.meeting.attendees if engine.meeting else [])]
    names += [e.name for e in (engine.brief.entities if engine.brief else [])]
    names = list(dict.fromkeys(n for n in names if n))
    return ("Участники и темы: " + ", ".join(names))[:200] if names else None


def speaker_label(engine) -> str:
    attendees = engine.meeting.attendees if engine.meeting else []
    return attendees[0].name if len(attendees) == 1 else "Собеседник"


def live_driver(rt: Runtime):
    from .capture import build_capture

    async def drive() -> None:
        asr = build_asr(rt.settings.asr)
        rt.bus.publish("warning", message="Загружаю модель распознавания…")
        await asyncio.to_thread(asr.warmup)
        pipeline = AsrPipeline(
            asr,
            rt.settings.asr,
            rt.engine.feed,
            prompt=lambda: asr_prompt(rt.engine),
            speaker=lambda: speaker_label(rt.engine),
            on_error=lambda exc: rt.bus.publish("error", message=f"Распознавание: {type(exc).__name__}: {exc}"),
        )
        await pipeline.run(build_capture(rt.settings.capture))

    return drive


def run_windowed(session) -> int:
    """GUI must own the main thread, so the asyncio session runs in a worker thread."""
    from .ui.macos import run_window

    ready, box = threading.Event(), {}

    def worker() -> None:
        async def main() -> None:
            box["loop"], box["stop"] = asyncio.get_running_loop(), asyncio.Event()
            await session(on_ready=lambda url: (box.update(url=url), ready.set()), stop=box["stop"])

        try:
            asyncio.run(main())
        except Exception as exc:
            box["error"] = exc
        finally:
            ready.set()

    thread = threading.Thread(target=worker, daemon=True, name="session")
    thread.start()
    if not ready.wait(120) or "url" not in box:
        print(f"Окно не открыто: {box.get('error', 'HUD не поднялся')}", file=sys.stderr)
        return 1
    run_window(box["url"])
    deadline = time.monotonic() + 180
    while thread.is_alive() and time.monotonic() < deadline:
        box["loop"].call_soon_threadsafe(box["stop"].set)
        thread.join(0.5)
    return 1 if "error" in box else 0


# --- commands ---------------------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    return doctor.render(asyncio.run(doctor.run_checks(args.config, live=not args.no_live)))


def cmd_replay(args: argparse.Namespace) -> int:
    settings = load(args)
    data_dir = args.data_dir or tempfile.mkdtemp(prefix="callc-replay-")
    settings = settings.model_copy(update={"privacy": settings.privacy.model_copy(update={"data_dir": str(data_dir)})})
    clock = VirtualClock()
    rt = build_runtime(settings, llm=args.llm, kernel=SnapshotKernel(args.kernel) if args.kernel else None, clock=clock, ephemeral=True)
    utterances = load_transcript(args.transcript)

    async def driver() -> None:
        await replay(rt.engine, utterances, clock, speed=args.speed, settle=args.speed == 0)

    opts = SessionOptions(
        meeting=build_meeting(args), mode=args.mode, ui=args.ui, ticker=False, auto_approve=args.yes, summarize=not args.no_summary
    )
    outcome = asyncio.run(run_session(rt, opts, driver))
    Console().print(f"Файлы сессии: {outcome.session_dir}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    console = Console()
    settings = load(args)
    if args.llm == "claude" and not has_credentials():
        console.print("[red]Нет доступа к Claude.[/] Задайте ANTHROPIC_API_KEY или выполните ant auth login (или запустите с --llm fake).")
        return 2
    rt = build_runtime(settings, llm=args.llm, kernel=SnapshotKernel(args.kernel) if args.kernel else None)
    removed = rt.store.purge()
    if removed:
        console.print(f"Удалено старых сессий: {removed}")

    async def session(on_ready=None, stop=None):
        meeting = await choose_meeting(rt, args, console)
        opts = SessionOptions(
            meeting=meeting,
            mode=args.mode,
            ui=args.window or not args.no_ui,
            terminal=not args.no_terminal,
            compact=args.compact or args.window,
            open_browser=args.open,
            summarize=not args.no_summary,
            auto_approve=args.yes,
            on_ready=on_ready,
        )
        return await run_session(rt, opts, live_driver(rt), stop=stop, console=console)

    if args.window:
        return run_windowed(session)
    asyncio.run(session())
    return 0


def cmd_brief(args: argparse.Namespace) -> int:
    settings = load(args)
    meeting = build_meeting(args)
    if meeting is None:
        print("Укажите --meeting, либо --title / --attendee", file=sys.stderr)
        return 2
    kernel = SnapshotKernel(args.kernel) if args.kernel else build_kernel(settings)

    async def fetch():
        try:
            return await kernel.brief(meeting, max_sensitivity=PolicyEngine(settings.policy, settings.privacy).sensitivity_limit(args.mode or settings.privacy.mode))
        finally:
            await kernel.aclose()

    brief = asyncio.run(fetch())
    console = Console()
    table = Table("источник", "дата", "заголовок", "фрагмент", title=f"Досье: {meeting.title}", show_lines=False)
    for item in brief.items:
        table.add_row(item.source, item.ts.strftime("%d.%m.%y") if item.ts else "", escape(item.title), escape(item.text[:70]))
    console.print(table)
    for heading, rows in (("Хвосты", brief.open_loops), ("Тезисы", brief.talking_points)):
        for row in rows:
            console.print(f"[bold]{heading}:[/] {escape(row)}")
    console.print(f"Сущности: {', '.join(e.name for e in brief.entities) or '—'}")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    from .llm import build_llm

    session = Path(args.session).expanduser()
    transcript = session / "transcript.jsonl"
    if not transcript.exists():
        print("Расшифровка не сохранялась: включите privacy.store_transcripts = true.", file=sys.stderr)
        return 2
    settings = load(args)
    meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
    actions_file = session / "actions.json"
    record = CallRecord(
        session_id=meta["session_id"],
        meeting=Meeting(id=meta["session_id"], title=meta["title"]),
        started_at=meta["started_at"],
        utterances=[Utterance.model_validate_json(line) for line in transcript.read_text(encoding="utf-8").splitlines() if line.strip()],
        action_items=[ActionItem.model_validate(a) for a in json.loads(actions_file.read_text(encoding="utf-8"))] if actions_file.exists() else [],
    )
    summary = asyncio.run(summarize_call(record, build_llm(args.llm, settings), settings, PolicyEngine(settings.policy, settings.privacy)))
    print(summary.model_dump_json(indent=2))
    return 0


def cmd_pack_info(args: argparse.Namespace) -> int:
    path = Path(args.pack).expanduser()
    items, entities, meetings, skipped = read_pack(path)
    console = Console()
    console.print(f"{path}: {path.stat().st_size / 1024:.0f} КБ, материалов {len(items)}, сущностей {len(entities)}, встреч {len(meetings)}, пропущено строк {skipped}")
    for label, counter in (
        ("по источникам", Counter(i.source for i in items)),
        ("по чувствительности", Counter(i.sensitivity for i in items)),
        ("по видам", Counter(i.kind for i in items)),
    ):
        console.print(f"  {label}: " + ", ".join(f"{k} {v}" for k, v in counter.most_common()))
    dated = sorted(i.ts for i in items if i.ts)
    if dated:
        console.print(f"  период: {dated[0]:%d.%m.%Y} — {dated[-1]:%d.%m.%Y}")
    return 1 if skipped else 0


def cmd_demo_kernel(args: argparse.Namespace) -> int:
    from .kernel_server import serve

    items, entities, meetings, _ = read_pack(args.pack)
    token = args.token or secrets.token_urlsafe(18)
    print(f"Демо-kernel на http://{args.host}:{args.port}\nexport KERNEL_COMPANION_TOKEN={token}")
    serve(MemoryDataSource(items, entities, meetings), token, args.host, args.port)
    return 0


def cmd_audit_verify(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser() if args.path else load(args).data_dir / "audit.jsonl"
    if not path.exists():
        print(f"Журнала нет: {path}")
        return 0
    ok, broken = AuditLog(path).verify()
    print("Цепочка цела." if ok else f"Нарушена на записи №{broken}.")
    return 0 if ok else 1


def cmd_sessions_purge(args: argparse.Namespace) -> int:
    settings = load(args)
    removed = SessionStore(settings.data_dir, retention_days=settings.privacy.retention_days).purge()
    print(f"Удалено сессий: {removed}")
    return 0


# --- parser ---------------------------------------------------------------------------------------------


def _meeting_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--meeting", help="JSON с описанием встречи (см. examples/meeting.json)")
    parser.add_argument("--title", help="название встречи, если файла нет")
    parser.add_argument("--attendee", action="append", help="участник «Имя» или «Имя:Организация», можно несколько раз")
    parser.add_argument("--agenda", help="повестка")
    parser.add_argument("--mode", choices=MODES, help="режим разговора (по умолчанию из конфига)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="callc", description="Живой компаньон для созвонов")
    parser.add_argument("--config", help="путь к config.toml (по умолчанию ~/.config/callcompanion/config.toml)")
    sub = parser.add_subparsers(dest="command", required=True)

    doc = sub.add_parser("doctor", help="проверить, готов ли этот компьютер к звонку")
    doc.add_argument("--no-live", action="store_true", help="не проверять звук и распознавание")
    doc.set_defaults(func=cmd_doctor)

    rep = sub.add_parser("replay", help="прогнать записанный звонок через движок (демо, без микрофона)")
    rep.add_argument("transcript", help="JSONL с репликами (см. examples/sample_call.jsonl)")
    rep.add_argument("--kernel", help="файл пакета контекста (pack.jsonl)")
    rep.add_argument("--llm", choices=["fake", "claude"], default="fake")
    rep.add_argument("--speed", type=float, default=0.0, help="0 — мгновенно, 1 — в реальном времени")
    rep.add_argument("--user", help="как вас зовут в разговоре")
    rep.add_argument("--ui", action="store_true", help="поднять HUD и держать его открытым")
    rep.add_argument("--yes", action="store_true", help="сразу подтвердить записи в kernel")
    rep.add_argument("--no-summary", action="store_true")
    rep.add_argument("--data-dir", help="куда класть файлы сессии (по умолчанию временная папка)")
    _meeting_args(rep)
    rep.set_defaults(func=cmd_replay)

    run = sub.add_parser("run", help="слушать живой звонок")
    run.add_argument("--llm", choices=["claude", "fake"], default="claude")
    run.add_argument("--kernel", help="файл пакета контекста вместо настроек kernel")
    run.add_argument("--user", help="как вас зовут в разговоре")
    run.add_argument("--meeting-id", help="взять встречу из kernel по id")
    run.add_argument("--auto", action="store_true", help="дождаться встречи из календаря")
    run.add_argument("--no-ui", action="store_true", help="без HUD, только консоль")
    run.add_argument("--window", action="store_true", help="плавающее окно HUD (macOS, extras [mac])")
    run.add_argument("--open", action="store_true", help="открыть HUD в браузере")
    run.add_argument("--compact", action="store_true", help="компактный HUD для узкого окна")
    run.add_argument("--no-terminal", action="store_true")
    run.add_argument("--no-summary", action="store_true")
    run.add_argument("--yes", action="store_true", help="после звонка сразу подтвердить записи в kernel")
    _meeting_args(run)
    run.set_defaults(func=cmd_run)

    brief = sub.add_parser("brief", help="показать досье перед звонком")
    brief.add_argument("--kernel", help="файл пакета контекста вместо настроек kernel")
    _meeting_args(brief)
    brief.set_defaults(func=cmd_brief)

    summ = sub.add_parser("summarize", help="заново оформить итоги сохранённого звонка")
    summ.add_argument("session", help="папка сессии с transcript.jsonl")
    summ.add_argument("--llm", choices=["claude", "fake"], default="claude")
    summ.set_defaults(func=cmd_summarize)

    pack = sub.add_parser("pack-info", help="проверить файл пакета контекста")
    pack.add_argument("pack")
    pack.set_defaults(func=cmd_pack_info)

    demo = sub.add_parser("demo-kernel", help="поднять демо-сервер kernel на примере пакета, чтобы проверить сеть")
    demo.add_argument("--pack", required=True)
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=8765)
    demo.add_argument("--token")
    demo.set_defaults(func=cmd_demo_kernel)

    audit = sub.add_parser("audit", help="журнал действий")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    verify = audit_sub.add_parser("verify", help="проверить цепочку хэшей")
    verify.add_argument("--path")
    verify.set_defaults(func=cmd_audit_verify)

    sessions = sub.add_parser("sessions", help="сохранённые сессии")
    sessions_sub = sessions.add_subparsers(dest="sessions_command", required=True)
    purge = sessions_sub.add_parser("purge", help="удалить сессии старше срока хранения")
    purge.set_defaults(func=cmd_sessions_purge)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    except (FileNotFoundError, KernelError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
