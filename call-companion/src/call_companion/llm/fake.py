"""Deterministic stand-in for tests and the demo: it phrases the top evidence, nothing smarter."""

from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator

from ..models import CallSummary, Note
from ..util import clip
from .base import SummaryRequest, WhisperRequest


class FakeLLM:
    def __init__(self, latency_s: float = 0.0, chunk: int = 14) -> None:
        self.latency_s = latency_s
        self.chunk = chunk
        self.requests: list[WhisperRequest] = []

    def _compose(self, req: WhisperRequest) -> str:
        if not req.evidence:
            return "ASK|Уточните, пожалуйста, о чём именно вопрос?" if req.trigger_kind in ("stall", "manual") else "SKIP"
        label, top = req.evidence[0]
        sentence = clip(re.split(r"(?<=[.!?])\s", top.text.strip())[0], 140)
        if req.trigger_kind in ("stall", "question", "manual"):
            kind = "ANSWER"
        else:
            kind = "REMIND" if top.kind == "open_loop" else "FACT"
        return f"{kind}|{sentence} [{label}]"

    async def whisper(self, req: WhisperRequest) -> AsyncIterator[str]:
        self.requests.append(req)
        line = self._compose(req)
        pieces = [line[i : i + self.chunk] for i in range(0, len(line), self.chunk)]
        for piece in pieces:
            if self.latency_s:
                await asyncio.sleep(self.latency_s / len(pieces))
            yield piece

    async def summarize(self, req: SummaryRequest) -> CallSummary:
        facts = [Note(text=c.text, tags=["созвон"], meeting_id=req.meeting.id) for c in req.action_candidates]
        agreed = "; ".join(c.text for c in req.action_candidates) or "договорённостей не зафиксировано"
        return CallSummary(
            title=req.meeting.title or "Созвон",
            tldr=f"Реплик в записи: {len(req.transcript.splitlines())}.",
            action_items=list(req.action_candidates),
            facts_to_remember=facts,
            followup_draft=f"Спасибо за созвон! Фиксирую: {agreed}.",
        )
