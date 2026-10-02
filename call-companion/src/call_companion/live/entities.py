"""Spot known people, companies and projects in speech, tolerant to Slavic inflections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..models import Entity
from ..util import STOPWORDS, name_stem, tokenize

_DIGRAPHS = [
    ("shch", "щ"), ("sch", "щ"), ("sh", "ш"), ("ch", "ч"), ("zh", "ж"), ("kh", "х"), ("ts", "ц"),
    ("ya", "я"), ("yu", "ю"), ("yo", "ё"), ("ph", "ф"), ("th", "т"), ("ck", "к"), ("ee", "и"), ("oo", "у"),
]
_LETTERS = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х", "i": "и", "j": "дж",
    "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т",
    "u": "у", "v": "в", "w": "в", "x": "кс", "y": "й", "z": "з",
}


def latin_to_cyrillic(text: str) -> str:
    """Rough phonetic transliteration: ASR in Russian mode writes foreign names in Cyrillic."""
    out = text.lower()
    for latin, cyrillic in _DIGRAPHS:
        out = out.replace(latin, cyrillic)
    return "".join(_LETTERS.get(ch, ch) for ch in out)


@dataclass(frozen=True)
class EntityMatch:
    entity: Entity
    surface: str


@dataclass(frozen=True)
class _Pattern:
    entity: Entity
    stems: tuple[str, ...]
    any_token: bool


class EntityIndex:
    def __init__(self, entities: Iterable[Entity] = ()) -> None:
        self.entities = list(entities)
        self._patterns: list[_Pattern] = []
        for entity in self.entities:
            for name in self._names(entity):
                stems = tuple(dict.fromkeys(name_stem(t) for t in tokenize(name) if t not in STOPWORDS and len(t) >= 3))
                if stems:
                    self._patterns.append(_Pattern(entity, stems, any_token=entity.kind == "person"))

    @staticmethod
    def _names(entity: Entity) -> list[str]:
        names = [entity.name, *entity.aliases]
        names += [latin_to_cyrillic(n) for n in names if n.isascii()]
        return list(dict.fromkeys(n for n in names if n.strip()))

    def __bool__(self) -> bool:
        return bool(self._patterns)

    def find(self, text: str) -> list[EntityMatch]:
        tokens = tokenize(text)
        stems = [name_stem(t) for t in tokens]
        found: dict[str, EntityMatch] = {}
        for pattern in self._patterns:
            if pattern.entity.name in found:
                continue
            if pattern.any_token:
                hits = [i for i, s in enumerate(stems) if s in pattern.stems and len(tokens[i]) >= 4]
            elif all(s in stems for s in pattern.stems):
                hits = [stems.index(s) for s in pattern.stems]
            else:
                hits = []
            if hits:
                surface = " ".join(tokens[i] for i in sorted(set(hits)))
                found[pattern.entity.name] = EntityMatch(pattern.entity, surface)
        return list(found.values())
