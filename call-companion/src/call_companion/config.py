"""Settings loaded from ~/.config/callcompanion/config.toml; every key is optional."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from .models import MeetingMode, Model

DEFAULT_CONFIG_PATH = Path("~/.config/callcompanion/config.toml").expanduser()
Effort = Literal["low", "medium", "high", "xhigh", "max"]


class ConfigError(Exception):
    pass


class UserCfg(Model):
    name: str = ""
    aliases: list[str] = Field(default_factory=list)
    language: str = "auto"


class LlmCfg(Model):
    model: str = "claude-opus-5-5"
    effort: Effort = "low"
    max_tokens: int = 1024
    timeout_s: float = 20.0
    summary_model: str = "claude-opus-5-5"
    summary_effort: Effort = "medium"
    summary_max_tokens: int = 8000
    fallbacks: bool = True


class KernelHttpCfg(Model):
    base_url: str = ""
    token_env: str = "KERNEL_COMPANION_TOKEN"
    timeout_s: float = 0.8
    brief_timeout_s: float = 8.0


class KernelCliCfg(Model):
    command: list[str] = Field(default_factory=lambda: ["kernel", "companion"])
    timeout_s: float = 6.0


class KernelCfg(Model):
    backends: list[Literal["snapshot", "http", "cli"]] = Field(default_factory=lambda: ["snapshot", "http"])
    snapshot_path: str = "~/.callcompanion/pack.jsonl"
    search_budget_s: float = 0.9
    http: KernelHttpCfg = Field(default_factory=KernelHttpCfg)
    cli: KernelCliCfg = Field(default_factory=KernelCliCfg)


class LiveCfg(Model):
    assist: Literal["always", "on_hesitation", "manual"] = "on_hesitation"
    hesitation_s: float = 3.0
    max_whispers_per_min: int = 4
    min_gap_s: float = 8.0
    context_window_s: float = 90.0
    entity_cooldown_s: float = 120.0
    multi_party_requires_name: bool = True
    max_evidence: int = 5


class AsrCfg(Model):
    engine: Literal["mlx", "faster"] = "mlx"
    model: str = "mlx-community/whisper-large-v3-turbo"
    language: str = "auto"
    min_speech_s: float = 0.4
    max_segment_s: float = 12.0
    end_silence_s: float = 0.55


class CaptureCfg(Model):
    backend: Literal["sounddevice", "audiotap"] = "sounddevice"
    mic_device: str | None = None
    system_device: str = "BlackHole 2ch"
    audiotap_path: str = "~/.callcompanion/bin/audiotap"


class UiCfg(Model):
    host: str = "127.0.0.1"
    port: int = 7777
    token: str = ""
    allowed_hosts: list[str] = Field(default_factory=list)


class PrivacyCfg(Model):
    mode: MeetingMode = "client"
    store_transcripts: bool = False
    retention_days: int = 14
    redact: bool = True
    extra_redact_patterns: list[str] = Field(default_factory=list)
    data_dir: str = "~/.callcompanion"


class PolicyCfg(Model):
    auto_allow: list[str] = Field(default_factory=list)
    approval_ttl_s: float = 1800.0
    mode_limits: dict[str, str] = Field(default_factory=dict)


class Settings(Model):
    user: UserCfg = Field(default_factory=UserCfg)
    llm: LlmCfg = Field(default_factory=LlmCfg)
    kernel: KernelCfg = Field(default_factory=KernelCfg)
    live: LiveCfg = Field(default_factory=LiveCfg)
    asr: AsrCfg = Field(default_factory=AsrCfg)
    capture: CaptureCfg = Field(default_factory=CaptureCfg)
    ui: UiCfg = Field(default_factory=UiCfg)
    privacy: PrivacyCfg = Field(default_factory=PrivacyCfg)
    policy: PolicyCfg = Field(default_factory=PolicyCfg)

    @property
    def data_dir(self) -> Path:
        return Path(self.privacy.data_dir).expanduser()

    @property
    def user_aliases(self) -> list[str]:
        names = [self.user.name, *self.user.aliases]
        return list(dict.fromkeys(n.strip() for n in names if n.strip()))

    def kernel_token(self) -> str | None:
        return os.environ.get(self.kernel.http.token_env) or None


def load_settings(path: str | Path | None = None) -> Settings:
    candidate = Path(path).expanduser() if path else Path(os.environ.get("CALLC_CONFIG", DEFAULT_CONFIG_PATH))
    if not candidate.exists():
        if path:
            raise ConfigError(f"Конфиг не найден: {candidate}")
        return Settings()
    try:
        raw = tomllib.loads(candidate.read_text(encoding="utf-8"))
        return Settings.model_validate(raw)
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ConfigError(f"Ошибка в конфиге {candidate}: {exc}") from exc
