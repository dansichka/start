"""Offline kernel: a JSONL pack exported by the kernel host and synced to this machine.

Each line is {"type": "item" | "entity" | "meeting", ...fields}; a missing type means item.
Notes are never written into the pack; they go to a sibling outbox file for the kernel to import.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from ..kernel_api import MemoryDataSource, read_pack
from ..models import Brief, ContextItem, Health, Meeting, Note, SearchFilters
from ..util import new_id
from .base import BaseKernel, KernelUnavailable


class SnapshotKernel(BaseKernel):
    name = "snapshot"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.outbox_path = self.path.with_name(self.path.name + ".outbox.jsonl")
        self.skipped = 0
        self._mtime: int | None = None
        self._source = MemoryDataSource()
        self._lock = asyncio.Lock()

    async def _ready(self) -> MemoryDataSource:
        try:
            mtime = self.path.stat().st_mtime_ns
        except FileNotFoundError as exc:
            raise KernelUnavailable(f"snapshot: нет файла {self.path}") from exc
        if mtime != self._mtime:
            async with self._lock:
                if mtime != self._mtime:
                    self._source = await asyncio.to_thread(self._load)
                    self._mtime = mtime
        return self._source

    def _load(self) -> MemoryDataSource:
        items, entities, meetings, self.skipped = read_pack(self.path)
        return MemoryDataSource(items, entities, meetings)

    async def health(self) -> Health:
        source = await self._ready()
        return Health(ok=True, version="snapshot", items=len(source.items))

    async def search(self, query: str, *, k: int = 6, filters: SearchFilters | None = None) -> list[ContextItem]:
        return (await self._ready()).search(query, k, filters or SearchFilters())

    async def brief(self, meeting: Meeting, *, max_sensitivity: str = "work") -> Brief:
        return (await self._ready()).brief(meeting, max_sensitivity)

    async def upcoming(self, within_min: int = 30) -> list[Meeting]:
        return (await self._ready()).upcoming(within_min)

    async def save_note(self, note: Note) -> str:
        note_id = new_id("note")
        record = {"type": "note", "id": note_id, "created_at": datetime.now(timezone.utc).isoformat(), **note.model_dump(mode="json")}
        self.outbox_path.parent.mkdir(parents=True, exist_ok=True)
        with self.outbox_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return note_id
