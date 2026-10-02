"""Data models shared by the companion and the kernel contract."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator


def _to_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


UtcDatetime = Annotated[datetime, AfterValidator(_to_utc)]
Channel = Literal["me", "them"]
MeetingMode = Literal["internal", "client", "interview", "public"]
SuggestionKind = Literal["answer", "fact", "reminder", "ask", "warn"]

SENSITIVITY_RANK: dict[str, int] = {"public": 0, "work": 1, "private": 2, "secret": 3}


def sensitivity_rank(label: str) -> int:
    """Unknown labels rank as secret so a typo on the kernel side never widens access."""
    return SENSITIVITY_RANK.get(label, SENSITIVITY_RANK["secret"])


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Utterance(Model):
    id: str
    channel: Channel
    speaker: str
    text: str
    t_start: float
    t_end: float
    final: bool = True
    lang: str | None = None


class ContextItem(Model):
    id: str
    source: str
    kind: str = "note"
    title: str = ""
    text: str = ""
    ts: UtcDatetime | None = None
    url: str | None = None
    entities: list[str] = Field(default_factory=list)
    sensitivity: str = "work"
    score: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sensitivity", mode="before")
    @classmethod
    def _known_label(cls, value: Any) -> str:
        label = str(value or "work").lower()
        return label if label in SENSITIVITY_RANK else "secret"


class Entity(Model):
    name: str
    kind: str = "other"
    aliases: list[str] = Field(default_factory=list)
    summary: str = ""
    item_ids: list[str] = Field(default_factory=list)


class Participant(Model):
    name: str
    email: str | None = None
    org: str | None = None
    aliases: list[str] = Field(default_factory=list)


class Meeting(Model):
    id: str
    title: str = ""
    start: UtcDatetime | None = None
    end: UtcDatetime | None = None
    attendees: list[Participant] = Field(default_factory=list)
    agenda: str = ""
    platform: str | None = None
    mode: MeetingMode | None = None
    language: str | None = None


class Brief(Model):
    meeting: Meeting
    items: list[ContextItem] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    talking_points: list[str] = Field(default_factory=list)
    open_loops: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utcnow)


class SearchFilters(Model):
    sources: list[str] = Field(default_factory=list)
    since: UtcDatetime | None = None
    entities: list[str] = Field(default_factory=list)
    max_sensitivity: str = "work"


class Health(Model):
    ok: bool
    version: str = ""
    items: int | None = None
    latency_ms: float | None = None


class Note(Model):
    text: str
    tags: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    meeting_id: str | None = None
    source: str = "call-companion"
    untrusted: bool = True


class ProposedAction(Model):
    id: str
    capability: str
    title: str
    payload: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    origin: Literal["llm", "user", "system"] = "llm"
    created_at: datetime = Field(default_factory=utcnow)


class Suggestion(Model):
    id: str
    kind: SuggestionKind
    text: str
    evidence: list[str] = Field(default_factory=list)
    trigger: str = ""
    created_at: float = 0.0
    ttl: float = 45.0
    final: bool = True


class ActionItem(Model):
    id: str
    text: str
    owner: str | None = None
    due: str | None = None
    status: Literal["candidate", "confirmed", "dismissed"] = "candidate"
    utterance_id: str | None = None


class CallSummary(Model):
    title: str = ""
    tldr: str = ""
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    facts_to_remember: list[Note] = Field(default_factory=list)
    followup_draft: str = ""


class CallRecord(Model):
    session_id: str
    meeting: Meeting
    started_at: datetime
    ended_at: datetime | None = None
    utterances: list[Utterance] = Field(default_factory=list)
    suggestions: list[Suggestion] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
