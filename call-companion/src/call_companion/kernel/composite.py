"""Fan-out over several kernel backends with a hard time budget per call."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from ..models import Brief, ContextItem, Entity, Health, Meeting, Note, ProposedAction, SearchFilters
from .base import BaseKernel, KernelClient, KernelError, KernelUnavailable

log = logging.getLogger(__name__)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class CompositeKernel(BaseKernel):
    """Reads merge every backend that answers in time; writes go to the first backend that accepts.

    Backends are listed local-first (for example snapshot, then http): reads are merged, while
    writes try them in reverse so the live kernel is preferred and the local outbox is the fallback.
    """

    name = "composite"

    def __init__(self, backends: list[KernelClient], search_budget_s: float = 0.9, brief_budget_s: float = 10.0) -> None:
        if not backends:
            raise ValueError("нужен хотя бы один backend")
        self.backends = backends
        self.search_budget_s = search_budget_s
        self.brief_budget_s = brief_budget_s

    async def _gather(self, make_call, budget_s: float) -> list:
        async def guarded(backend: KernelClient):
            return await asyncio.wait_for(make_call(backend), budget_s)

        results = await asyncio.gather(*(guarded(b) for b in self.backends), return_exceptions=True)
        answers = []
        problems = []
        for backend, result in zip(self.backends, results):
            if isinstance(result, BaseException):
                reason = "таймаут" if isinstance(result, asyncio.TimeoutError) else str(result) or type(result).__name__
                problems.append(f"{backend.name}: {reason}")
            else:
                answers.append(result)
        if not answers:
            raise KernelUnavailable("; ".join(problems))
        if problems:
            log.info("kernel: часть источников недоступна: %s", "; ".join(problems))
        return answers

    async def health(self) -> Health:
        answers = await self._gather(lambda b: b.health(), self.search_budget_s * 2)
        return Health(
            ok=any(h.ok for h in answers),
            version="+".join(h.version for h in answers if h.version),
            items=sum(h.items or 0 for h in answers),
        )

    async def search(self, query: str, *, k: int = 6, filters: SearchFilters | None = None) -> list[ContextItem]:
        answers = await self._gather(lambda b: b.search(query, k=k, filters=filters), self.search_budget_s)
        merged: dict[str, ContextItem] = {}
        for items in answers:
            top = max((i.score for i in items), default=0.0) or 1.0
            for item in items:
                normalized = round(item.score / top, 4)
                known = merged.get(item.id)
                if known is None or normalized > known.score:
                    merged[item.id] = item.model_copy(update={"score": normalized})
        return sorted(merged.values(), key=lambda i: i.score, reverse=True)[:k]

    async def brief(self, meeting: Meeting, *, max_sensitivity: str = "work") -> Brief:
        answers = await self._gather(lambda b: b.brief(meeting, max_sensitivity=max_sensitivity), self.brief_budget_s)
        items: dict[str, ContextItem] = {}
        entities: dict[str, Entity] = {}
        talking: list[str] = []
        loops: list[str] = []
        for brief in answers:
            for item in brief.items:
                items.setdefault(item.id, item)
            for entity in brief.entities:
                entities.setdefault(entity.name, entity)
            talking += [t for t in brief.talking_points if t not in talking]
            loops += [t for t in brief.open_loops if t not in loops]
        return Brief(
            meeting=meeting,
            items=list(items.values()),
            entities=list(entities.values()),
            talking_points=talking,
            open_loops=loops,
        )

    async def upcoming(self, within_min: int = 30) -> list[Meeting]:
        answers = await self._gather(lambda b: b.upcoming(within_min), self.search_budget_s * 3)
        meetings: dict[str, Meeting] = {}
        for batch in answers:
            for meeting in batch:
                meetings.setdefault(meeting.id, meeting)
        return sorted(meetings.values(), key=lambda m: m.start or _EPOCH)

    async def save_note(self, note: Note) -> str:
        return await self._first_writer(lambda b: b.save_note(note))

    async def submit_action(self, action: ProposedAction) -> str:
        return await self._first_writer(lambda b: b.submit_action(action))

    async def _first_writer(self, make_call) -> str:
        problems = []
        for backend in reversed(self.backends):
            try:
                return await asyncio.wait_for(make_call(backend), self.brief_budget_s)
            except (KernelError, asyncio.TimeoutError) as exc:
                problems.append(f"{backend.name}: {exc or 'таймаут'}")
        raise KernelUnavailable("; ".join(problems))

    async def aclose(self) -> None:
        await asyncio.gather(*(b.aclose() for b in self.backends), return_exceptions=True)
