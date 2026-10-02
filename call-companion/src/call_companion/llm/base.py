"""LLM interface the engine depends on."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

from ..models import ActionItem, Brief, CallSummary, ContextItem, Meeting


class LLMError(Exception):
    pass


class LLMRefused(LLMError):
    """The model declined; the engine stays silent rather than show half an answer."""


@dataclass
class WhisperRequest:
    user_name: str
    language: str
    mode: str
    meeting: Meeting | None
    brief: Brief | None
    evidence: list[tuple[str, ContextItem]]
    transcript: str
    trigger_kind: str
    trigger_text: str
    stall_s: float = 0.0


@dataclass
class SummaryRequest:
    user_name: str
    language: str
    meeting: Meeting
    transcript: str
    action_candidates: list[ActionItem] = field(default_factory=list)
    brief: Brief | None = None


class LLM(Protocol):
    def whisper(self, req: WhisperRequest) -> AsyncIterator[str]: ...

    async def summarize(self, req: SummaryRequest) -> CallSummary: ...
