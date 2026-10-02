"""Decide when the companion should look something up: questions, hesitation, names, promises."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from ..config import LiveCfg
from ..models import Entity, Utterance
from ..util import name_stem, tokenize
from .entities import EntityIndex

QUESTION_WORDS = frozenset(
    """
    как что почему зачем когда где куда откуда сколько какой какая какое какие каком какого кто кого чей
    можешь можете сможешь сможете расскажи расскажите объясни объясните скажи скажите подскажи подскажите покажи покажите
    як що чому навіщо коли де куди звідки скільки який яка яке які хто можеш розкажи розкажіть поясни поясніть підкажи підкажіть
    how what why when where which who whom whose can could would will do does did is are was were have has should shall tell explain walk show
    """.split()
)
LEADING_FILLERS = frozenset("а и ну так слушай слушайте окей ок хорошо тогда а_вот і та добре well so and okay ok then".split())
FILLERS = frozenset("ну эм ээ э так угу ага мм хм тааак um uh hmm well так-так ну-у".split())

_PROMISE = re.compile(
    r"\b(?:я|мы)\s+(?:тебе\s+|вам\s+)?(?:пришлю|пришлём|пришлем|отправлю|отправим|сделаю|сделаем|подготовлю|подготовим|вышлю|вышлем"
    r"|созвонюсь|свяжусь|напишу|проверю|уточню|скину|вернусь|согласую)"
    r"|\b(?:я|ми)\s+(?:тобі\s+|вам\s+)?(?:надішлю|надішлемо|зроблю|зробимо|підготую|підготуємо|зв'яжусь|напишу|перевірю|уточню|скину|повернусь|погоджу)"
    r"|\b(?:i|we)(?:'ll|\s+will|\s+can|\s+shall)\s+(?:send|share|prepare|follow\s+up|check|review|schedule|draft|update|get\s+back|confirm)"
    r"|\blet\s+me\s+(?:send|check|look|get|confirm|share)",
    re.IGNORECASE,
)
_DEADLINE = re.compile(
    r"\b(?:до|к|ко)\s+(?:конца\s+(?:дня|недели|месяца)|понедельник\w*|вторник\w*|сред\w+|четверг\w*|пятниц\w*|суббот\w*|воскресень\w*|завтра|\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?|\d{1,2}\s+\w+)"
    r"|\bдо\s+(?:кінця\s+(?:дня|тижня|місяця)|понеділк\w*|вівтор\w*|серед\w+|четвер\w*|п'ятниц\w*|субот\w*|неділ\w*|завтра)"
    r"|\bby\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|eod|end\s+of\s+(?:day|week)|tomorrow|next\s+week|\d{1,2}[./]\d{1,2})"
    r"|\bдедлайн\w*|\bdeadline",
    re.IGNORECASE,
)


class TriggerKind(str, Enum):
    QUESTION = "question"
    STALL = "stall"
    ENTITY = "entity"
    MANUAL = "manual"
    COMMITMENT = "commitment"


@dataclass
class Commitment:
    phrase: str
    owner: str
    due: str | None = None


@dataclass
class Trigger:
    kind: TriggerKind
    utterance: Utterance | None = None
    question: str = ""
    entities: list[Entity] = field(default_factory=list)
    commitment: Commitment | None = None

    @property
    def directed(self) -> bool:
        return self.kind in (TriggerKind.QUESTION, TriggerKind.STALL, TriggerKind.MANUAL)


@dataclass
class _Pending:
    utterance: Utterance
    due: float
    entities: list[Entity]


def is_question(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if "?" in stripped:
        return True
    if re.search(r"[.!…]\s*$", stripped):
        return False
    tokens = tokenize(stripped)
    while tokens and tokens[0] in LEADING_FILLERS:
        tokens = tokens[1:]
    return len(tokens) >= 3 and tokens[0] in QUESTION_WORDS


def is_filler(text: str) -> bool:
    tokens = tokenize(text)
    return 0 < len(tokens) <= 3 and all(t in FILLERS for t in tokens)


def find_commitment(text: str, owner: str) -> Commitment | None:
    promise = _PROMISE.search(text)
    if not promise:
        return None
    deadline = _DEADLINE.search(text)
    return Commitment(phrase=promise.group(0), owner=owner, due=deadline.group(0) if deadline else None)


class TriggerDetector:
    def __init__(
        self,
        cfg: LiveCfg,
        user_aliases: list[str],
        entities: EntityIndex | None = None,
        remote_speakers: Callable[[], int] = lambda: 1,
    ) -> None:
        self.cfg = cfg
        self.entities = entities or EntityIndex()
        self._alias_stems = {name_stem(t) for alias in user_aliases for t in tokenize(alias) if len(t) >= 3}
        self._remote_speakers = remote_speakers
        self._pending: _Pending | None = None
        self._parked: tuple[_Pending, float] | None = None
        self._entity_seen: dict[str, float] = {}

    def set_entities(self, index: EntityIndex) -> None:
        self.entities = index

    @property
    def pending(self) -> _Pending | None:
        return self._pending

    def on_utterance(self, utterance: Utterance, now: float) -> list[Trigger]:
        if utterance.channel == "me":
            return self._on_me(utterance, now)
        if not utterance.final or not utterance.text.strip():
            return []
        return self._on_them(utterance, now)

    def tick(self, now: float) -> list[Trigger]:
        pending = self._pending
        if pending is None or now < pending.due:
            return []
        self._pending = None
        self._parked = None
        return [
            Trigger(TriggerKind.STALL, pending.utterance, pending.utterance.text, pending.entities)
        ]

    def manual(self, text: str = "", utterance: Utterance | None = None) -> Trigger:
        return Trigger(TriggerKind.MANUAL, utterance, text.strip())

    def _on_me(self, utterance: Utterance, now: float) -> list[Trigger]:
        triggers: list[Trigger] = []
        if utterance.final:
            commitment = find_commitment(utterance.text, "me")
            if commitment:
                triggers.append(Trigger(TriggerKind.COMMITMENT, utterance, commitment=commitment))
            if is_filler(utterance.text):
                parked = self._parked
                if parked and now - parked[1] < 8.0:
                    parked[0].due = now + self.cfg.hesitation_s
                    self._pending, self._parked = parked[0], None
                return triggers
            self._parked = None
        elif self._pending is not None:
            self._parked = (self._pending, now)
        self._pending = None
        return triggers

    def _on_them(self, utterance: Utterance, now: float) -> list[Trigger]:
        triggers: list[Trigger] = []
        commitment = find_commitment(utterance.text, utterance.speaker)
        if commitment:
            triggers.append(Trigger(TriggerKind.COMMITMENT, utterance, commitment=commitment))
        if self.cfg.assist == "manual":
            return triggers

        mentioned = [match.entity for match in self.entities.find(utterance.text)]
        fresh = self._mark_seen(mentioned, now)
        if is_question(utterance.text) and self._addressed(utterance):
            if self.cfg.assist == "always":
                triggers.append(Trigger(TriggerKind.QUESTION, utterance, utterance.text, mentioned))
            else:
                self._pending = _Pending(utterance, now + self.cfg.hesitation_s, mentioned)
            return triggers
        if self._pending is not None and utterance.speaker == self._pending.utterance.speaker:
            self._pending.due = now + self.cfg.hesitation_s
            self._pending.entities += [e for e in mentioned if e not in self._pending.entities]
            return triggers
        if fresh:
            triggers.append(Trigger(TriggerKind.ENTITY, utterance, utterance.text, fresh))
        return triggers

    def _mark_seen(self, mentioned: list[Entity], now: float) -> list[Entity]:
        """Record every mention; return only those not heard within the cooldown (for ambient triggers)."""
        fresh: list[Entity] = []
        for entity in mentioned:
            seen = self._entity_seen.get(entity.name)
            if seen is None or now - seen >= self.cfg.entity_cooldown_s:
                fresh.append(entity)
            self._entity_seen[entity.name] = now
        return fresh

    def _addressed(self, utterance: Utterance) -> bool:
        if self._remote_speakers() <= 1 or not self.cfg.multi_party_requires_name:
            return True
        return any(name_stem(t) in self._alias_stems for t in tokenize(utterance.text))
