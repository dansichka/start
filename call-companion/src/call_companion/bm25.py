"""Small BM25 index over ContextItem used for offline packs, brief lookups and tests."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from math import log

from .models import ContextItem, SearchFilters, sensitivity_rank
from .util import STOPWORDS, names_match, stem, tokenize

RECENCY_BOOST = 0.25
RECENCY_HALF_LIFE_DAYS = 21.0


def terms(text: str) -> list[str]:
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS]


def passes(item: ContextItem, filters: SearchFilters | None) -> bool:
    if item.sensitivity == "secret":
        return False
    if filters is None:
        return True
    if sensitivity_rank(item.sensitivity) > sensitivity_rank(filters.max_sensitivity):
        return False
    if filters.sources and item.source not in filters.sources:
        return False
    if filters.since and item.ts and item.ts < filters.since:
        return False
    if filters.entities and not any(names_match(a, b) for a in filters.entities for b in item.entities):
        return False
    return True


class Bm25Index:
    def __init__(self, k1: float = 1.4, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._items: list[ContextItem] = []
        self._tf: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._df: Counter[str] = Counter()
        self._avg = 1.0

    def __len__(self) -> int:
        return len(self._items)

    def extend(self, items: list[ContextItem]) -> None:
        for item in items:
            self.add(item)

    def add(self, item: ContextItem) -> None:
        weighted = terms(item.title) * 2 + terms(item.text) + terms(" ".join(item.entities)) * 2
        tf = Counter(weighted)
        self._items.append(item)
        self._tf.append(tf)
        self._lengths.append(max(len(weighted), 1))
        self._df.update(tf.keys())
        self._avg = sum(self._lengths) / len(self._lengths)

    def search(
        self,
        query: str,
        k: int = 6,
        filters: SearchFilters | None = None,
        now: datetime | None = None,
    ) -> list[ContextItem]:
        query_terms = list(dict.fromkeys(terms(query)))
        if not query_terms or not self._items:
            return []
        now = now or datetime.now(timezone.utc)
        total = len(self._items)
        scored: list[tuple[float, int]] = []
        for index, item in enumerate(self._items):
            if not passes(item, filters):
                continue
            tf, length = self._tf[index], self._lengths[index]
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = log(1 + (total - self._df[term] + 0.5) / (self._df[term] + 0.5))
                score += idf * freq * (self.k1 + 1) / (freq + self.k1 * (1 - self.b + self.b * length / self._avg))
            if score > 0:
                scored.append((score * self._recency(item, now), index))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [self._items[i].model_copy(update={"score": round(s, 4)}) for s, i in scored[:k]]

    @staticmethod
    def _recency(item: ContextItem, now: datetime) -> float:
        if item.ts is None:
            return 1.0
        age_days = max((now - item.ts).total_seconds() / 86400, 0.0)
        return 1.0 + RECENCY_BOOST * 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
