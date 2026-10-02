"""Server half of the kernel contract.

Drop this module into the kernel environment, implement DataSource on top of the kernel's own
storage, then expose it over a CLI (run_cli), as an offline pack (write_pack) or over HTTP via
kernel_server.create_app. This module needs nothing but pydantic.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

import logging

from pydantic import BaseModel, Field, ValidationError

from .bm25 import Bm25Index, passes
from .models import Brief, ContextItem, Entity, Meeting, Note, ProposedAction, SearchFilters
from .util import new_id, names_match

log = logging.getLogger(__name__)
MAX_K = 20
OPS = ("health", "search", "brief", "upcoming", "note", "action")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class UnknownOp(ValueError):
    pass


class DataSource(Protocol):
    """What the kernel implements; every method is synchronous and may block."""

    def stats(self) -> dict[str, Any]: ...

    def search(self, query: str, k: int, filters: SearchFilters) -> list[ContextItem]: ...

    def brief(self, meeting: Meeting, max_sensitivity: str) -> Brief: ...

    def upcoming(self, within_min: int) -> list[Meeting]: ...

    def save_note(self, note: Note) -> str: ...

    def run_action(self, action: ProposedAction) -> str: ...


class SearchRequest(BaseModel):
    query: str
    k: int = 6
    filters: SearchFilters = Field(default_factory=SearchFilters)


class BriefRequest(BaseModel):
    meeting: Meeting
    max_sensitivity: str = "work"


class UpcomingRequest(BaseModel):
    within_min: int = 30


def _enforce(items: list[ContextItem], max_sensitivity: str) -> list[ContextItem]:
    """Defence in depth: the server never returns more than the caller allowed, nor any secret."""
    limit = SearchFilters(max_sensitivity=max_sensitivity)
    return [item for item in items if passes(item, limit)]


def dispatch(source: DataSource, op: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Single implementation of every operation, shared by the HTTP and CLI transports."""
    if op == "health":
        return {"ok": True, **source.stats()}
    if op == "search":
        request = SearchRequest.model_validate(payload)
        k = min(max(request.k, 1), MAX_K)
        items = _enforce(source.search(request.query, k, request.filters), request.filters.max_sensitivity)
        return {"items": [item.model_dump(mode="json") for item in items[:k]]}
    if op == "brief":
        request = BriefRequest.model_validate(payload)
        brief = source.brief(request.meeting, request.max_sensitivity)
        brief.items = _enforce(brief.items, request.max_sensitivity)
        return brief.model_dump(mode="json")
    if op == "upcoming":
        request = UpcomingRequest.model_validate(payload)
        meetings = source.upcoming(min(max(request.within_min, 1), 24 * 60))
        return {"meetings": [meeting.model_dump(mode="json") for meeting in meetings]}
    if op == "note":
        return {"id": source.save_note(Note.model_validate(payload))}
    if op == "action":
        return {"id": source.run_action(ProposedAction.model_validate(payload))}
    raise UnknownOp(op)


def run_cli(source: DataSource, argv: list[str] | None = None) -> int:
    """`kernel companion <op>`: JSON request on stdin, JSON response on stdout, UTF-8 always."""
    for stream in (sys.stdin, sys.stdout):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] not in OPS:
        print(f"usage: <command> {{{'|'.join(OPS)}}} < request.json", file=sys.stderr)
        return 2
    raw = sys.stdin.read().strip()
    try:
        result = dispatch(source, args[0], json.loads(raw) if raw else {})
    except (ValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    return 0


def write_pack(
    path: str | Path,
    items: Iterable[ContextItem],
    entities: Iterable[Entity] = (),
    meetings: Iterable[Meeting] = (),
) -> int:
    """Write an offline pack atomically so a syncing laptop never reads a half-written file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    count = 0
    with temp.open("w", encoding="utf-8") as handle:
        for kind, rows in (("item", items), ("entity", entities), ("meeting", meetings)):
            for row in rows:
                handle.write(json.dumps({"type": kind, **row.model_dump(mode="json")}, ensure_ascii=False) + "\n")
                count += 1
    os.replace(temp, target)
    return count


def read_pack(path: str | Path) -> tuple[list[ContextItem], list[Entity], list[Meeting], int]:
    """Parse an offline pack; broken lines are skipped and counted instead of failing the whole file."""
    items: list[ContextItem] = []
    entities: list[Entity] = []
    meetings: list[Meeting] = []
    skipped = 0
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            kind = row.pop("type", "item")
            if kind == "item":
                items.append(ContextItem.model_validate(row))
            elif kind == "entity":
                entities.append(Entity.model_validate(row))
            elif kind == "meeting":
                meetings.append(Meeting.model_validate(row))
            else:
                raise ValueError(f"unknown type {kind!r}")
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            skipped += 1
            log.warning("pack %s:%d пропущена: %s", Path(path).name, number, exc)
    return items, entities, meetings, skipped


def build_brief(
    index: Bm25Index,
    items: list[ContextItem],
    entities: list[Entity],
    meeting: Meeting,
    max_sensitivity: str,
    limit: int = 24,
) -> Brief:
    """Reference brief: everything linked to the attendees first, then BM25 on title and agenda."""
    filters = SearchFilters(max_sensitivity=max_sensitivity)
    names = [n for p in meeting.attendees for n in (p.name, *p.aliases, p.org or "") if n]
    linked = sorted(
        (i for i in items if passes(i, filters) and any(names_match(n, e) for n in names for e in i.entities)),
        key=lambda i: i.ts or _EPOCH,
        reverse=True,
    )
    chosen: dict[str, ContextItem] = {i.id: i for i in linked[:limit]}
    for item in index.search(" ".join([meeting.title, meeting.agenda, *names]), k=limit, filters=filters):
        chosen.setdefault(item.id, item)
    picked = list(chosen.values())[:limit]
    mentioned = {name for item in picked for name in item.entities}
    relevant = [
        e
        for e in entities
        if e.name in mentioned or any(names_match(n, a) for n in names for a in (e.name, *e.aliases))
    ]
    return Brief(
        meeting=meeting,
        items=picked,
        entities=relevant[:30],
        talking_points=[i.text for i in picked if i.kind == "talking_point"],
        open_loops=[i.text for i in picked if i.kind == "open_loop"],
    )


class MemoryDataSource:
    """Reference DataSource over plain lists; used by the snapshot client, tests and the demo."""

    def __init__(
        self,
        items: Iterable[ContextItem] = (),
        entities: Iterable[Entity] = (),
        meetings: Iterable[Meeting] = (),
    ) -> None:
        self.items = list(items)
        self.entities = list(entities)
        self.meetings = list(meetings)
        self.notes: list[Note] = []
        self.actions: list[ProposedAction] = []
        self._index = Bm25Index()
        self._index.extend(self.items)

    def stats(self) -> dict[str, Any]:
        return {"version": "memory", "items": len(self.items)}

    def search(self, query: str, k: int, filters: SearchFilters) -> list[ContextItem]:
        return self._index.search(query, k, filters)

    def brief(self, meeting: Meeting, max_sensitivity: str) -> Brief:
        return build_brief(self._index, self.items, self.entities, meeting, max_sensitivity)

    def upcoming(self, within_min: int) -> list[Meeting]:
        now = datetime.now(timezone.utc)
        window = (now - timedelta(minutes=5), now + timedelta(minutes=within_min))
        return [m for m in self.meetings if m.start and window[0] <= m.start <= window[1]]

    def save_note(self, note: Note) -> str:
        self.notes.append(note)
        return new_id("note")

    def run_action(self, action: ProposedAction) -> str:
        self.actions.append(action)
        return action.id
