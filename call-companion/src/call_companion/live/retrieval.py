"""Pull evidence for a trigger from the pre-call brief and the live kernel, then keep only what is relevant."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Iterable

from ..bm25 import Bm25Index
from ..kernel.base import KernelClient, KernelError
from ..models import Brief, ContextItem, Entity, MeetingMode, SearchFilters
from ..policy import PolicyEngine
from ..util import keywords, name_stem, names_match, stem, tokenize


@dataclass
class Evidence:
    labels: dict[str, ContextItem] = field(default_factory=dict)
    latency_ms: float = 0.0
    degraded: bool = False
    query: str = ""

    @property
    def items(self) -> list[ContextItem]:
        return list(self.labels.values())


def _normalized(items: list[ContextItem]) -> list[ContextItem]:
    top = max((i.score for i in items), default=0.0) or 1.0
    return [i.model_copy(update={"score": round(i.score / top, 4)}) for i in items]


class ContextRetriever:
    def __init__(
        self,
        kernel: KernelClient,
        policy: PolicyEngine,
        *,
        mode: MeetingMode,
        budget_s: float = 0.9,
        max_evidence: int = 5,
        ignore: Iterable[str] = (),
    ) -> None:
        self.kernel = kernel
        self.policy = policy
        self.mode = mode
        self.budget_s = budget_s
        self.max_evidence = max_evidence
        self._ignore = {stem(t) for name in ignore for t in tokenize(name)} | {name_stem(t) for name in ignore for t in tokenize(name)}
        self._index = Bm25Index()

    def set_brief(self, brief: Brief | None) -> None:
        """Brief items must already be filtered and redacted by the policy."""
        self._index = Bm25Index()
        if brief:
            self._index.extend(brief.items)

    async def retrieve(self, focus: str, entities: Iterable[Entity] = (), context: str = "") -> Evidence:
        started = time.monotonic()
        entities = list(entities)
        names = [e.name for e in entities]
        extra = [t for t in keywords(context) if stem(t) not in self._ignore][:6]
        query = " ".join([focus, *names, *extra]).strip()
        wanted = {stem(t) for t in keywords(" ".join([focus, *names]))} - self._ignore
        if not query:
            return Evidence(query=query)

        candidates = {i.id: i for i in _normalized(self._index.search(query, k=self.max_evidence * 3))}
        degraded = False
        try:
            remote = await asyncio.wait_for(
                self.kernel.search(
                    query,
                    k=self.max_evidence * 3,
                    filters=SearchFilters(max_sensitivity=self.policy.sensitivity_limit(self.mode)),
                ),
                self.budget_s,
            )
            for item in _normalized(self.policy.filter_items(remote, self.mode)):
                known = candidates.get(item.id)
                if known is None or item.score > known.score:
                    candidates[item.id] = item
        except (KernelError, asyncio.TimeoutError):
            degraded = True

        kept = [i for i in candidates.values() if self._relevant(i, wanted, entities)]
        kept.sort(key=lambda i: (self._linked(i, entities), i.score), reverse=True)
        labels = {f"E{n}": item for n, item in enumerate(kept[: self.max_evidence], 1)}
        return Evidence(labels, round((time.monotonic() - started) * 1000, 1), degraded, query)

    @staticmethod
    def _linked(item: ContextItem, entities: list[Entity]) -> bool:
        return any(names_match(e.name, name) for e in entities for name in item.entities)

    def _relevant(self, item: ContextItem, wanted: set[str], entities: list[Entity]) -> bool:
        if self._linked(item, entities):
            return True
        if not wanted:
            return False
        present = {stem(t) for t in keywords(f"{item.title} {item.text} {' '.join(item.entities)}")}
        hits = len(wanted & present)
        return hits >= 2 or hits / len(wanted) >= 0.34
