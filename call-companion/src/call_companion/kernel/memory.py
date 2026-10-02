"""In-process kernel for tests and the demo, with injectable latency and failures."""

from __future__ import annotations

import asyncio

from ..kernel_api import MemoryDataSource
from ..models import Brief, ContextItem, Health, Meeting, Note, ProposedAction, SearchFilters
from .base import BaseKernel, KernelUnavailable


class InMemoryKernel(BaseKernel):
    name = "memory"

    def __init__(self, source: MemoryDataSource | None = None, *, latency_s: float = 0.0, fail: bool = False) -> None:
        self.source = source or MemoryDataSource()
        self.latency_s = latency_s
        self.fail = fail
        self.calls: list[str] = []

    async def _enter(self, op: str) -> None:
        self.calls.append(op)
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        if self.fail:
            raise KernelUnavailable("memory: injected failure")

    async def health(self) -> Health:
        await self._enter("health")
        return Health(ok=True, version="memory", items=len(self.source.items))

    async def search(self, query: str, *, k: int = 6, filters: SearchFilters | None = None) -> list[ContextItem]:
        await self._enter("search")
        return self.source.search(query, k, filters or SearchFilters())

    async def brief(self, meeting: Meeting, *, max_sensitivity: str = "work") -> Brief:
        await self._enter("brief")
        return self.source.brief(meeting, max_sensitivity)

    async def upcoming(self, within_min: int = 30) -> list[Meeting]:
        await self._enter("upcoming")
        return self.source.upcoming(within_min)

    async def save_note(self, note: Note) -> str:
        await self._enter("note")
        return self.source.save_note(note)

    async def submit_action(self, action: ProposedAction) -> str:
        await self._enter("action")
        return self.source.run_action(action)
