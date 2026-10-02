import os
import socket
import time
import types
from pathlib import Path

import pytest

from call_companion import doctor
from call_companion.config import Settings
from call_companion.doctor import FAIL, OK, WARN
from call_companion.policy import AuditLog

from .support import EXAMPLES


def settings_with(tmp_path, **sections) -> Settings:
    base = {"privacy": {"data_dir": str(tmp_path / "data")}, "kernel": {"backends": ["snapshot"], "snapshot_path": str(tmp_path / "pack.jsonl")}}
    for key, value in sections.items():
        base[key] = {**base.get(key, {}), **value}
    return Settings.model_validate(base)


def test_python_and_platform():
    assert doctor.check_python((3, 10)).status == FAIL and doctor.check_python((3, 12)).status == OK
    assert doctor.check_platform("Darwin", "arm64").status == OK
    intel = doctor.check_platform("Darwin", "x86_64")
    assert intel.status == WARN and "faster" in intel.hint
    assert doctor.check_platform("Linux", "x86_64").status == WARN


def test_credentials_from_env_a_profile_or_nothing(tmp_path):
    assert doctor.check_credentials({"ANTHROPIC_API_KEY": "k"}, tmp_path).status == OK
    assert doctor.check_credentials({"ANTHROPIC_AUTH_TOKEN": "t"}, tmp_path).status == OK
    assert doctor.check_credentials({}, tmp_path).status == FAIL
    (tmp_path / ".config" / "anthropic").mkdir(parents=True)
    assert "профиль" in doctor.check_credentials({}, tmp_path).detail


def test_config_states(tmp_path):
    missing, _ = doctor.check_config(str(tmp_path / "none.toml"))
    assert missing.status == FAIL
    os.environ.pop("CALLC_CONFIG", None)
    good = tmp_path / "c.toml"
    good.write_text('[user]\nname = "Даня"\n', encoding="utf-8")
    ok, settings = doctor.check_config(str(good))
    assert ok.status == OK and settings.user.name == "Даня"
    bad = tmp_path / "bad.toml"
    bad.write_text('[live]\nassist = "x"\n', encoding="utf-8")
    assert doctor.check_config(str(bad))[0].status == FAIL


def test_default_config_location_is_a_warning_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CALLC_CONFIG", str(tmp_path / "absent.toml"))
    check, settings = doctor.check_config(None)
    assert check.status == WARN and settings.llm.model == "claude-opus-5-5"


def test_user_name_check():
    assert doctor.check_user(Settings()).status == WARN
    assert doctor.check_user(Settings.model_validate({"user": {"name": "Даня"}})).status == OK


def test_data_dir_is_private_and_unwritable_locations_fail(tmp_path):
    ok = doctor.check_data_dir(settings_with(tmp_path))
    assert ok.status == OK and oct((tmp_path / "data").stat().st_mode & 0o777) == "0o700"
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    assert doctor.check_data_dir(settings_with(tmp_path, privacy={"data_dir": str(blocker / "sub")})).status == FAIL


def test_audit_chain_check(tmp_path):
    settings = settings_with(tmp_path)
    assert doctor.check_audit(settings).status == OK
    log = AuditLog(tmp_path / "data" / "audit.jsonl")
    log.append("a")
    log.append("b")
    assert doctor.check_audit(settings).detail == "цепочка хэшей цела"
    path = tmp_path / "data" / "audit.jsonl"
    path.write_text(path.read_text(encoding="utf-8").splitlines()[1] + "\n", encoding="utf-8")
    assert doctor.check_audit(settings).status == FAIL


async def test_kernel_check_states(tmp_path):
    import shutil

    pack = tmp_path / "pack.jsonl"
    settings = settings_with(tmp_path)
    missing = await doctor.check_kernel(settings)
    assert missing.status == FAIL and "kernel companion export" in missing.hint
    shutil.copy(EXAMPLES / "sample_kernel_pack.jsonl", pack)
    fresh = await doctor.check_kernel(settings)
    assert fresh.status == OK and "материалов 9" in fresh.detail
    old = time.time() - 3 * 86400
    os.utime(pack, (old, old))
    stale = await doctor.check_kernel(settings)
    assert stale.status == WARN and "ч" in stale.detail and "export" in stale.hint
    nothing = settings_with(tmp_path, kernel={"backends": ["http"]})
    assert (await doctor.check_kernel(nothing)).status == FAIL


def test_asr_check_states(tmp_path, monkeypatch):
    settings = settings_with(tmp_path)
    monkeypatch.setitem(__import__("sys").modules, "mlx_whisper", None)
    broken = doctor.check_asr(settings, tmp_path)
    assert broken.status == FAIL and "call-companion[mac]" in broken.hint
    monkeypatch.setitem(__import__("sys").modules, "mlx_whisper", types.ModuleType("mlx_whisper"))
    assert doctor.check_asr(settings, tmp_path).status == WARN
    (tmp_path / ".cache" / "huggingface" / "hub" / "models--mlx-community--whisper-large-v3-turbo").mkdir(parents=True)
    assert doctor.check_asr(settings, tmp_path).status == OK


def test_audio_check_states(tmp_path):
    tap = tmp_path / "audiotap"
    settings = settings_with(tmp_path, capture={"backend": "audiotap", "audiotap_path": str(tap)})
    assert doctor.check_audio(settings).status == FAIL
    tap.write_text("#!/bin/sh\n", encoding="utf-8")
    tap.chmod(0o755)
    assert doctor.check_audio(settings).status == OK

    sd = lambda *names: types.SimpleNamespace(query_devices=lambda: [{"name": n, "max_input_channels": 2} for n in names])
    default = settings_with(tmp_path)
    assert doctor.check_audio(default, sd("MacBook Microphone", "BlackHole 2ch")).status == OK
    missing = doctor.check_audio(default, sd("MacBook Microphone"))
    assert missing.status == FAIL and "BlackHole" in missing.hint
    crashed = types.SimpleNamespace(query_devices=lambda: (_ for _ in ()).throw(OSError("no portaudio")))
    assert doctor.check_audio(default, crashed).status == FAIL


def test_port_and_resources(tmp_path):
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        busy = settings_with(tmp_path, ui={"port": taken.getsockname()[1]})
        assert doctor.check_ui_port(busy).status == WARN
    assert doctor.check_ui_port(settings_with(tmp_path, ui={"port": 0})).status == OK
    settings = settings_with(tmp_path)
    assert doctor.check_resources(settings, memory_gb=16, free_gb=50).status == OK
    assert doctor.check_resources(settings, memory_gb=8, free_gb=50).status == WARN
    assert "small" in doctor.check_resources(settings, memory_gb=8, free_gb=50).hint
    assert doctor.check_resources(settings, memory_gb=16, free_gb=2).status == WARN


def test_render_summarises_and_sets_the_exit_code():
    lines = []
    checks = [doctor.Check("a", OK, "хорошо"), doctor.Check("b", WARN, "так себе", "сделайте X"), doctor.Check("c", FAIL, "сломано", "сделайте Y")]
    assert doctor.render(checks, lines.append) == 1
    text = "\n".join(lines)
    assert " ✓ a: хорошо" in text and " ! b: так себе" in text and "→ сделайте Y" in text and "1 в порядке, 1 предупреждений, 1 ошибок" in text
    assert doctor.render(checks[:2], lambda _: None) == 0
