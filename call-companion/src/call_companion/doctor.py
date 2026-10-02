"""`callc doctor`: what works, what does not and the exact fix, checked on the machine that will take the call."""

from __future__ import annotations

import asyncio
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import ConfigError, Settings, load_settings
from .policy import AuditLog

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    hint: str = ""


def check_python(version: tuple[int, int] | None = None) -> Check:
    version = version or sys.version_info[:2]
    text = f"{version[0]}.{version[1]}"
    if version >= (3, 11):
        return Check("Python", OK, text)
    return Check("Python", FAIL, text, "нужен Python 3.11+: brew install python@3.12 или uv python install 3.12")


def check_platform(system: str | None = None, machine: str | None = None) -> Check:
    system, machine = system or platform.system(), machine or platform.machine()
    label = f"{system} {machine}"
    if system == "Darwin" and machine == "arm64":
        return Check("Платформа", OK, label + " (Apple Silicon)")
    if system == "Darwin":
        return Check("Платформа", WARN, label, "Intel-Mac: MLX недоступен, поставьте extras [faster] и asr.engine = \"faster\"")
    return Check("Платформа", WARN, label, "захват системного звука рассчитан на macOS; здесь работают replay и тесты")


def check_credentials(env: dict[str, str] | None = None, home: Path | None = None) -> Check:
    env = os.environ if env is None else env
    if env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN"):
        return Check("Доступ к Claude", OK, "ключ в переменной окружения")
    if ((home or Path.home()) / ".config" / "anthropic").exists():
        return Check("Доступ к Claude", OK, "профиль ant auth login")
    return Check("Доступ к Claude", FAIL, "нет ключа", "export ANTHROPIC_API_KEY=... (или ant auth login); без него работает только --llm fake")


def check_config(path: str | None) -> tuple[Check, Settings]:
    try:
        settings = load_settings(path)
    except ConfigError as exc:
        return Check("Конфиг", FAIL, str(exc), "исправьте значение или удалите ключ"), Settings()
    source = Path(path or os.environ.get("CALLC_CONFIG", "~/.config/callcompanion/config.toml")).expanduser()
    if source.exists():
        return Check("Конфиг", OK, str(source)), settings
    return Check("Конфиг", WARN, "файла нет, значения по умолчанию", f"скопируйте examples/config.example.toml в {source}"), settings


def check_user(settings: Settings) -> Check:
    if settings.user_aliases:
        return Check("Ваше имя", OK, ", ".join(settings.user_aliases))
    return Check("Ваше имя", WARN, "не задано", "в [user] укажите name и aliases: по ним понятно, что вопрос адресован вам")


def check_data_dir(settings: Settings) -> Check:
    path = settings.data_dir
    try:
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o700)
        probe = path / ".probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Check("Папка данных", FAIL, f"{path}: {exc}", "проверьте права или задайте privacy.data_dir")
    return Check("Папка данных", OK, f"{path} (права 700)")


def check_audit(settings: Settings) -> Check:
    path = settings.data_dir / "audit.jsonl"
    if not path.exists():
        return Check("Журнал действий", OK, "пока пуст")
    ok, broken = AuditLog(path).verify()
    if ok:
        return Check("Журнал действий", OK, "цепочка хэшей цела")
    return Check("Журнал действий", FAIL, f"запись №{broken} изменена или удалена", "журнал вели не вы? сохраните файл и разберитесь")


async def check_kernel(settings: Settings, timeout: float = 4.0) -> Check:
    from .kernel import KernelError, build_kernel

    try:
        kernel = build_kernel(settings)
    except KernelError as exc:
        return Check("kernel", FAIL, str(exc), "в [kernel] включите snapshot, http или cli")
    started = time.monotonic()
    try:
        health = await asyncio.wait_for(kernel.health(), timeout)
    except (KernelError, asyncio.TimeoutError) as exc:
        if "нет файла" in str(exc):
            return Check("kernel", FAIL, str(exc), "пакета ещё нет: выгрузите его на ПК (kernel companion export) и положите в snapshot_path, либо уберите snapshot из [kernel].backends")
        return Check("kernel", FAIL, str(exc) or "таймаут", "ПК включён? Tailscale поднят? токен верный? актуален ли pack.jsonl?")
    finally:
        await kernel.aclose()
    detail = f"{', '.join(settings.kernel.backends)}: материалов {health.items}, ответ {round((time.monotonic() - started) * 1000)} мс"
    pack = Path(settings.kernel.snapshot_path).expanduser()
    if "snapshot" in settings.kernel.backends and pack.exists():
        age_h = (time.time() - pack.stat().st_mtime) / 3600
        if age_h > 24:
            return Check("kernel", WARN, f"{detail}; пакету {age_h:.0f} ч", "выгрузите свежий пакет на ПК: kernel companion export")
    return Check("kernel", OK, detail)


def check_asr(settings: Settings, home: Path | None = None) -> Check:
    engine = settings.asr.engine
    module = "mlx_whisper" if engine == "mlx" else "faster_whisper"
    try:
        __import__(module)
    except ImportError:
        extra = "mac" if engine == "mlx" else "faster"
        return Check("Распознавание", FAIL, f"нет {module}", f"pip install 'call-companion[{extra}]'")
    if engine == "mlx":
        cache = (home or Path.home()) / ".cache" / "huggingface" / "hub" / ("models--" + settings.asr.model.replace("/", "--"))
        if not cache.exists():
            return Check("Распознавание", WARN, f"{settings.asr.model}: модель не скачана", "скачается при первом запуске (~1.6 ГБ); лучше заранее: callc run --llm fake")
    return Check("Распознавание", OK, f"{engine}: {settings.asr.model}")


def check_audio(settings: Settings, sd_module=None) -> Check:
    cfg = settings.capture
    if cfg.backend == "audiotap":
        path = Path(cfg.audiotap_path).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return Check("Захват звука", OK, f"audiotap: {path}")
        return Check("Захват звука", FAIL, f"нет {path}", "соберите native/macos/AudioTap (docs/MACOS_SETUP.md)")
    try:
        sd = sd_module or __import__("sounddevice")
        names = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
    except Exception as exc:
        return Check("Захват звука", FAIL, f"sounddevice: {exc}", "pip install 'call-companion[mac]'")
    if not any(cfg.system_device.lower() in n.lower() for n in names):
        return Check("Захват звука", FAIL, f"нет устройства «{cfg.system_device}»; есть: {', '.join(names) or '—'}", "поставьте BlackHole 2ch и настройте Multi-Output (docs/MACOS_SETUP.md)")
    return Check("Захват звука", OK, f"микрофон + «{cfg.system_device}»")


def check_ui_port(settings: Settings) -> Check:
    with socket.socket() as probe:
        try:
            probe.bind((settings.ui.host if settings.ui.host != "0.0.0.0" else "", settings.ui.port))
        except OSError as exc:
            return Check("Порт HUD", WARN, f"{settings.ui.port}: {exc}", "освободите порт или задайте ui.port")
    return Check("Порт HUD", OK, f"{settings.ui.host}:{settings.ui.port}")


def check_resources(settings: Settings, memory_gb: float | None = None, free_gb: float | None = None) -> Check:
    import psutil

    memory_gb = memory_gb if memory_gb is not None else psutil.virtual_memory().total / 2**30
    free_gb = free_gb if free_gb is not None else psutil.disk_usage(str(Path.home())).free / 2**30
    detail = f"ОЗУ {memory_gb:.0f} ГБ, свободно на диске {free_gb:.0f} ГБ"
    if free_gb < 5:
        return Check("Ресурсы", WARN, detail, "для моделей нужно ~5 ГБ свободного места")
    if memory_gb <= 8 and "large" in settings.asr.model:
        return Check("Ресурсы", WARN, detail, "на 8 ГБ рядом с видеозвонком тесно: asr.model = \"mlx-community/whisper-small-mlx\"")
    return Check("Ресурсы", OK, detail)


async def run_checks(config_path: str | None = None, *, live: bool = True) -> list[Check]:
    config_check, settings = check_config(config_path)
    checks = [check_python(), check_platform(), config_check, check_user(settings), check_credentials(), check_data_dir(settings), check_audit(settings)]
    checks.append(await check_kernel(settings))
    if live:
        checks += [check_asr(settings), check_audio(settings), check_ui_port(settings), check_resources(settings)]
    return checks


def render(checks: list[Check], say: Callable[[str], None] = print) -> int:
    marks = {OK: "✓", WARN: "!", FAIL: "✗"}
    for check in checks:
        say(f" {marks[check.status]} {check.name}: {check.detail}")
        if check.hint and check.status != OK:
            say(f"     → {check.hint}")
    failed = sum(c.status == FAIL for c in checks)
    warned = sum(c.status == WARN for c in checks)
    say(f"\nИтого: {len(checks) - failed - warned} в порядке, {warned} предупреждений, {failed} ошибок.")
    return 1 if failed else 0
