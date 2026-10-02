import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

from call_companion import cli
from call_companion.asr.base import AsrResult
from call_companion.capture.base import AudioFrame
from call_companion.cli import main
from call_companion.policy import AuditLog

from .support import EXAMPLES

CALL = str(EXAMPLES / "sample_call.jsonl")
MEETING = str(EXAMPLES / "meeting.json")
SRC = Path(__file__).parents[1] / "src"


@pytest.fixture
def pack(tmp_path) -> Path:
    path = tmp_path / "pack.jsonl"
    shutil.copy(EXAMPLES / "sample_kernel_pack.jsonl", path)
    return path


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CALLC_CONFIG", str(home / "none.toml"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    return home


def replay_args(pack, tmp_path, *extra) -> list[str]:
    return ["replay", CALL, "--kernel", str(pack), "--meeting", MEETING, "--user", "Даня", "--data-dir", str(tmp_path / "data"), *extra]


def test_replay_runs_the_demo_end_to_end(pack, tmp_path, capsys):
    assert main(replay_args(pack, tmp_path)) == 0
    out = capsys.readouterr().out
    for fragment in ("Досье: Статус Orion", "▌ ФАКТ", "▌ ОТВЕТ", "▌ НАПОМИНАНИЕ", "Итоги: Статус Orion", "Ждут вашего решения: 4", "followup.md", "Звонок завершён"):
        assert fragment in out, fragment
    (session,) = (tmp_path / "data" / "sessions").iterdir()
    assert (session / "summary.json").exists() and (session / "followup.md").exists()
    assert not pack.with_name("pack.jsonl.outbox.jsonl").exists(), "nothing is written without approval"


def test_replay_yes_flag_approves_and_the_outbox_receives_the_notes(pack, tmp_path, capsys):
    assert main(replay_args(pack, tmp_path, "--yes")) == 0
    assert "Подтверждено флагом --yes: 4" in capsys.readouterr().out
    rows = [json.loads(line) for line in pack.with_name("pack.jsonl.outbox.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and all(r["type"] == "note" and r["untrusted"] for r in rows)


def test_replay_without_summary_and_with_a_missing_file(pack, tmp_path, capsys):
    assert main(replay_args(pack, tmp_path, "--no-summary")) == 0
    (session,) = (tmp_path / "data" / "sessions").iterdir()
    assert not (session / "summary.json").exists()
    capsys.readouterr()
    assert main(["replay", str(tmp_path / "nope.jsonl")]) == 1
    assert "Ошибка" in capsys.readouterr().err


def test_replay_in_the_hud_mode_serves_the_page_while_it_runs(pack, tmp_path, monkeypatch, capsys):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = tmp_path / "c.toml"
    config.write_text(f"[ui]\nport = {port}\n", encoding="utf-8")
    seen = {}
    original = cli.run_session

    async def spy(rt, opts, driver, stop=None, console=None):
        async def checked() -> None:
            await driver()
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as http:
                seen["anonymous"] = (await http.get("/api/state")).status_code

        return await original(rt, opts, checked, stop, console)

    monkeypatch.setattr(cli, "run_session", spy)
    assert main(["--config", str(config), *replay_args(pack, tmp_path, "--ui", "--no-summary")]) == 0
    assert seen["anonymous"] == 401, "the HUD refuses requests without its token"
    assert f"HUD: http://127.0.0.1:{port}/?token=" in capsys.readouterr().out


def test_doctor_reports_a_healthy_setup_and_names_the_fix_for_a_broken_one(pack, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    config = tmp_path / "c.toml"
    config.write_text(
        f'[user]\nname = "Даня"\n[kernel]\nbackends = ["snapshot"]\nsnapshot_path = "{pack}"\n[privacy]\ndata_dir = "{tmp_path / "data"}"\n',
        encoding="utf-8",
    )
    assert main(["--config", str(config), "doctor", "--no-live"]) == 0
    out = capsys.readouterr().out
    assert "✓ Python" in out and "✓ kernel: snapshot: материалов 9" in out and "0 ошибок" in out
    pack.unlink()
    assert main(["--config", str(config), "doctor", "--no-live"]) == 1
    out = capsys.readouterr().out
    assert "✗ kernel" in out and "→ пакета ещё нет" in out


def test_pack_info_summarises_and_flags_broken_lines(pack, capsys):
    assert main(["pack-info", str(pack)]) == 0
    out = capsys.readouterr().out
    assert "материалов 9, сущностей 4, встреч 0, пропущено строк 0" in out and "telegram 2" in out and "secret 1" in out
    with pack.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
    assert main(["pack-info", str(pack)]) == 1
    assert "пропущено строк 1" in capsys.readouterr().out


def test_brief_prints_the_dossier(pack, capsys):
    assert main(["brief", "--kernel", str(pack), "--meeting", MEETING]) == 0
    out = capsys.readouterr().out
    assert "Договорённости по бюджету" in out and "Хвосты: Обещал Максу" in out and "Orion" in out
    assert "Выплата" not in out and "Личный чат" not in out, "private and secret items stay out in client mode"
    assert main(["brief", "--kernel", str(pack)]) == 2


def test_brief_can_build_a_meeting_from_flags(pack, capsys):
    assert main(["brief", "--kernel", str(pack), "--title", "Созвон", "--attendee", "Max Weber", "--attendee", "Олена Коваленко:Acme Logistics"]) == 0
    assert "Обещал Максу" in capsys.readouterr().out


def test_audit_verify_and_sessions_purge(tmp_path, capsys):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for event in ("a", "b"):
        log.append(event)
    assert main(["audit", "verify", "--path", str(path)]) == 0 and "цела" in capsys.readouterr().out
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(lines[1] + "\n", encoding="utf-8")
    assert main(["audit", "verify", "--path", str(path)]) == 1
    assert main(["audit", "verify", "--path", str(tmp_path / "missing.jsonl")]) == 0
    config = tmp_path / "c.toml"
    config.write_text(f'[privacy]\ndata_dir = "{tmp_path / "data"}"\nretention_days = 1\n', encoding="utf-8")
    old = tmp_path / "data" / "sessions" / "old"
    old.mkdir(parents=True)
    ancient = time.time() - 5 * 86400
    os.utime(old, (ancient, ancient))
    capsys.readouterr()
    assert main(["--config", str(config), "sessions", "purge"]) == 0
    assert "Удалено сессий: 1" in capsys.readouterr().out and not old.exists()


def test_summarize_works_from_a_stored_transcript(pack, tmp_path, capsys):
    config = tmp_path / "c.toml"
    config.write_text(f'[privacy]\nstore_transcripts = true\ndata_dir = "{tmp_path / "data"}"\n', encoding="utf-8")
    assert main(["--config", str(config), *replay_args(pack, tmp_path, "--no-summary")]) == 0
    (session,) = (tmp_path / "data" / "sessions").iterdir()
    capsys.readouterr()
    assert main(["--config", str(config), "summarize", str(session), "--llm", "fake"]) == 0
    assert json.loads(capsys.readouterr().out)["title"] == "Статус Orion"
    (session / "transcript.jsonl").unlink()
    assert main(["summarize", str(session), "--llm", "fake"]) == 2


def test_run_refuses_to_start_without_access_to_claude(pack, capsys):
    assert main(["run", "--kernel", str(pack), "--no-ui"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().out


def test_a_bad_config_is_reported_not_raised(tmp_path, capsys):
    config = tmp_path / "bad.toml"
    config.write_text('[live]\nassist = "sometimes"\n', encoding="utf-8")
    assert main(["--config", str(config), "pack-info", str(EXAMPLES / "sample_kernel_pack.jsonl")]) == 0
    assert main(["--config", str(config), "audit", "verify"]) == 2
    assert "Ошибка в конфиге" in capsys.readouterr().err


def test_no_command_is_a_usage_error():
    with pytest.raises(SystemExit) as stop:
        main([])
    assert stop.value.code == 2


# --- the live path with scripted audio and recognition -------------------------------------------


class ScriptedEngine:
    name = "scripted"

    def __init__(self, texts: list[str]) -> None:
        self.texts, self.prompts, self.warmed = list(texts), [], False

    def warmup(self) -> None:
        self.warmed = True

    def transcribe(self, audio, *, language, prompt):
        self.prompts.append(prompt)
        return AsrResult(text=self.texts.pop(0), language="ru")


class ScriptedSource:
    def __init__(self, frames: list[AudioFrame], linger: float = 0.6) -> None:
        self.frames_, self.linger = frames, linger

    async def frames(self):
        for frame in self.frames_:
            yield frame
            await asyncio.sleep(0)
        await asyncio.sleep(self.linger)


def audio_frames(channel: str, parts: list[np.ndarray]) -> list[AudioFrame]:
    raw = np.concatenate(parts).tobytes()
    return [AudioFrame(channel, raw[i : i + 3200], i / 2 / 16000) for i in range(0, len(raw), 3200)]


def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * 16000)) / 16000
    return (0.25 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)


def hush(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * 16000), dtype=np.int16)


def test_the_live_path_turns_scripted_speech_into_a_whisper_and_a_summary(pack, tmp_path, monkeypatch, capsys):
    engine = ScriptedEngine(["Давайте вернёмся к Orion и бюджету"])
    them = audio_frames("them", [hush(0.4), tone(1.5), hush(1.2)])
    monkeypatch.setattr(cli, "build_asr", lambda cfg: engine)
    monkeypatch.setattr("call_companion.capture.build_capture", lambda cfg: ScriptedSource(them))
    config = tmp_path / "c.toml"
    config.write_text(f'[user]\nname = "Даня"\n[privacy]\ndata_dir = "{tmp_path / "data"}"\n', encoding="utf-8")
    code = main(
        ["--config", str(config), "run", "--llm", "fake", "--no-ui", "--kernel", str(pack), "--title", "Статус Orion",
         "--attendee", "Олена Коваленко:Acme Logistics", "--agenda", "Бюджет"]
    )
    out = capsys.readouterr().out
    assert code == 0 and engine.warmed
    assert "Загружаю модель распознавания" in out and "Давайте вернёмся к Orion и бюджету" in out
    assert "▌ ФАКТ" in out and "Итоги: Статус Orion" in out
    assert engine.prompts[0].startswith("Участники и темы:") and "Олена Коваленко" in engine.prompts[0]


def test_a_capture_failure_is_reported_and_the_call_is_still_wrapped_up(pack, tmp_path, monkeypatch, capsys):
    from call_companion.capture.base import CaptureError

    class Broken:
        async def frames(self):
            raise CaptureError("нет устройства «BlackHole 2ch»")
            yield

    monkeypatch.setattr(cli, "build_asr", lambda cfg: ScriptedEngine([]))
    monkeypatch.setattr("call_companion.capture.build_capture", lambda cfg: Broken())
    config = tmp_path / "c.toml"
    config.write_text(f'[privacy]\ndata_dir = "{tmp_path / "data"}"\n', encoding="utf-8")
    assert main(["--config", str(config), "run", "--llm", "fake", "--no-ui", "--kernel", str(pack), "--title", "x"]) == 0
    out = capsys.readouterr().out
    assert "✗ CaptureError: нет устройства «BlackHole 2ch»" in out and "Звонок завершён" in out


def test_helpers_build_asr_hints_and_speaker_labels():
    from types import SimpleNamespace

    from call_companion.models import Brief, Entity, Meeting, Participant

    meeting = Meeting(id="m", attendees=[Participant(name="Олена Коваленко")])
    brief = Brief(meeting=meeting, entities=[Entity(name="Orion"), Entity(name="Олена Коваленко")])
    engine = SimpleNamespace(meeting=meeting, brief=brief)
    assert cli.asr_prompt(engine) == "Участники и темы: Олена Коваленко, Orion"
    assert cli.speaker_label(engine) == "Олена Коваленко"
    assert cli.speaker_label(SimpleNamespace(meeting=meeting.model_copy(update={"attendees": []}), brief=None)) == "Собеседник"
    assert cli.asr_prompt(SimpleNamespace(meeting=None, brief=None)) is None
    long = SimpleNamespace(meeting=None, brief=Brief(meeting=meeting, entities=[Entity(name=f"Имя{i}" * 5) for i in range(30)]))
    assert len(cli.asr_prompt(long)) == 200


async def test_choose_meeting_from_the_kernel_calendar(pack, capsys):
    from argparse import Namespace
    from datetime import datetime, timedelta, timezone

    from rich.console import Console

    from call_companion.app import build_runtime
    from call_companion.config import Settings
    from call_companion.kernel.memory import InMemoryKernel
    from call_companion.kernel_api import MemoryDataSource
    from call_companion.llm.fake import FakeLLM
    from call_companion.models import Meeting

    soon = Meeting(id="soon", title="Скоро", start=datetime.now(timezone.utc) + timedelta(minutes=2))
    rt = build_runtime(Settings(), llm=FakeLLM(), kernel=InMemoryKernel(MemoryDataSource(meetings=[soon])), ephemeral=True)
    args = lambda **kw: Namespace(meeting=None, title=None, attendee=None, agenda=None, meeting_id=None, auto=False, **kw)
    console = Console(file=open(os.devnull, "w"))
    assert (await cli.choose_meeting(rt, args(), console)) is None
    assert (await cli.choose_meeting(rt, args(), console)) is None
    chosen = await cli.choose_meeting(rt, Namespace(**{**vars(args()), "meeting_id": "soon"}), console)
    assert chosen.id == "soon"
    auto = await asyncio.wait_for(cli.choose_meeting(rt, Namespace(**{**vars(args()), "auto": True}), console), 3)
    assert auto.id == "soon"
    missing = await cli.choose_meeting(rt, Namespace(**{**vars(args()), "meeting_id": "nope"}), console)
    assert missing is None


# --- demo kernel as a real subprocess ---------------------------------------------------------------


def test_demo_kernel_serves_the_pack_over_http(pack):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    process = subprocess.Popen(
        [sys.executable, "-m", "call_companion", "demo-kernel", "--pack", str(pack), "--port", str(port), "--token", "demo-token"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        headers = {"Authorization": "Bearer demo-token"}
        for _ in range(100):
            try:
                health = httpx.get(f"http://127.0.0.1:{port}/v1/health", headers=headers, timeout=1)
                break
            except httpx.TransportError:
                time.sleep(0.1)
        else:
            pytest.fail("demo-kernel did not start: " + (process.stderr.read() if process.poll() is not None else ""))
        assert health.json()["ok"] is True and health.json()["items"] == 9
        assert httpx.get(f"http://127.0.0.1:{port}/v1/health", timeout=1).status_code == 401
        found = httpx.post(f"http://127.0.0.1:{port}/v1/search", headers=headers, json={"query": "бюджет этапа"}, timeout=2).json()
        assert found["items"][0]["id"] == "note:budget"
    finally:
        process.terminate()
        process.wait(timeout=5)
