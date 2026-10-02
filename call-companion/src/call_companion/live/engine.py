"""The live engine: utterances in, triggers evaluated, evidence retrieved, whispers streamed out."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Callable

from ..bus import EventBus
from ..config import Settings
from ..kernel.base import KernelClient, KernelError
from ..llm.base import LLM, LLMRefused, WhisperRequest
from ..llm.prompts import WhisperStream
from ..models import (
    ActionItem,
    Brief,
    CallRecord,
    Entity,
    Meeting,
    MeetingMode,
    ProposedAction,
    Suggestion,
    Utterance,
    utcnow,
)
from ..policy import PolicyEngine, Ticket
from ..util import clip, names_match, new_id
from .entities import EntityIndex
from .gate import WhisperGate, similar
from .retrieval import ContextRetriever
from .transcript import TranscriptBuffer
from .triggers import Trigger, TriggerDetector, TriggerKind

log = logging.getLogger(__name__)
PARTIAL_STEP_CHARS = 8
ERROR_REPORT_GAP_S = 10.0


class LiveEngine:
    def __init__(
        self,
        settings: Settings,
        kernel: KernelClient,
        llm: LLM,
        policy: PolicyEngine,
        bus: EventBus,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.settings = settings
        self.kernel = kernel
        self.llm = llm
        self.policy = policy
        self.bus = bus
        self.mode: MeetingMode = settings.privacy.mode
        self.session_id = new_id("call")
        self.meeting: Meeting | None = None
        self.brief: Brief | None = None
        self.suggestions: list[Suggestion] = []
        self.actions: list[ActionItem] = []
        self.transcript = TranscriptBuffer(settings.live.context_window_s)
        self.gate = WhisperGate(settings.live)
        self.detector = TriggerDetector(settings.live, settings.user_aliases, EntityIndex(), self._remote_count)
        self.retriever = self._new_retriever()
        self.started_at: datetime = utcnow()
        self._t0 = time.monotonic()
        self._clock = clock or (lambda: time.monotonic() - self._t0)
        self._speakers: set[str] = set()
        self._generation: asyncio.Task | None = None
        self._generation_manual = False
        self._tasks: set[asyncio.Task] = set()
        self._last_error_at = float("-inf")
        self._sources: dict[str, list[dict]] = {}
        self._ended = False
        self.summary: dict | None = None

    def now(self) -> float:
        return self._clock()

    def _new_retriever(self) -> ContextRetriever:
        return ContextRetriever(
            self.kernel,
            self.policy,
            mode=self.mode,
            budget_s=self.settings.kernel.search_budget_s,
            max_evidence=self.settings.live.max_evidence,
            ignore=self.settings.user_aliases,
        )

    def _remote_count(self) -> int:
        known = len(self.meeting.attendees) if self.meeting else 0
        return max(1, known, len(self._speakers))

    # --- lifecycle -------------------------------------------------------------

    async def start(self, meeting: Meeting | None = None, mode: MeetingMode | None = None) -> Brief:
        meeting = meeting or Meeting(id=new_id("adhoc"), title="Созвон")
        self.meeting = self._without_me(meeting)
        self.mode = mode or meeting.mode or self.settings.privacy.mode
        self.started_at = utcnow()
        self._t0 = time.monotonic()
        await self._load_brief()
        self._publish_status("listening")
        return self.brief

    def _without_me(self, meeting: Meeting) -> Meeting:
        mine = self.settings.user_aliases
        others = [p for p in meeting.attendees if not any(names_match(a, p.name) for a in mine)]
        return meeting.model_copy(update={"attendees": others})

    async def _load_brief(self) -> None:
        assert self.meeting is not None
        try:
            brief = await self.kernel.brief(self.meeting, max_sensitivity=self.policy.sensitivity_limit(self.mode))
        except KernelError as exc:
            self.bus.publish("warning", message=f"kernel недоступен, работаю без досье: {exc}")
            brief = Brief(meeting=self.meeting)
        brief.items = self.policy.filter_items(brief.items, self.mode)
        brief.open_loops = [self.policy.redact(t) for t in brief.open_loops]
        brief.talking_points = [self.policy.redact(t) for t in brief.talking_points]
        self.brief = brief

        known = {e.name: e for e in brief.entities}
        for person in self.meeting.attendees:
            known.setdefault(person.name, Entity(name=person.name, kind="person", aliases=person.aliases))
            if person.org:
                known.setdefault(person.org, Entity(name=person.org, kind="org"))
        self.detector.set_entities(EntityIndex(known.values()))
        self.retriever = self._new_retriever()
        self.retriever.set_brief(brief)
        self.bus.publish(
            "brief",
            title=self.meeting.title,
            attendees=[p.name for p in self.meeting.attendees],
            items=len(brief.items),
            open_loops=brief.open_loops,
            talking_points=brief.talking_points,
            mode=self.mode,
        )

    async def set_mode(self, mode: MeetingMode) -> None:
        """A stricter mode takes effect at once; a looser one waits for the brief it newly allows."""
        self._cancel_generation()
        self.mode = mode
        if self.brief is not None:
            self.brief = self.brief.model_copy(update={"items": self.policy.filter_items(self.brief.items, mode)})
        self.retriever = self._new_retriever()
        self.retriever.set_brief(self.brief)
        if self.meeting is not None:
            await self._load_brief()
        self._publish_status("listening")

    def mute(self, muted: bool) -> None:
        self.gate.muted = muted
        self._publish_status("listening")

    def panic(self) -> None:
        """Clear the HUD and stop pushing anything until unmuted."""
        self._cancel_generation()
        self.gate.muted = True
        self.bus.publish("clear")
        self._publish_status("listening")

    async def stop(self) -> CallRecord:
        self._cancel_generation()
        await self.settle()
        self._ended = True
        self.bus.publish("status", state="ended", mode=self.mode, muted=self.gate.muted)
        return CallRecord(
            session_id=self.session_id,
            meeting=self.meeting or Meeting(id=new_id("adhoc")),
            started_at=self.started_at,
            ended_at=utcnow(),
            utterances=self.transcript.finals,
            suggestions=list(self.suggestions),
            action_items=list(self.actions),
        )

    async def settle(self) -> None:
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # --- input -----------------------------------------------------------------

    def feed(self, utterance: Utterance) -> None:
        utterance = utterance.model_copy(update={"text": self.policy.redact(utterance.text)})
        if utterance.channel == "them" and utterance.final and utterance.text.strip():
            self._speakers.add(utterance.speaker)
        self.transcript.add(utterance)
        if utterance.text.strip():
            self.bus.publish("utterance", **utterance.model_dump())
        for trigger in self.detector.on_utterance(utterance, self.now()):
            if trigger.kind is TriggerKind.COMMITMENT:
                self._note_commitment(trigger)
            else:
                self._launch(trigger)

    def tick(self) -> None:
        for trigger in self.detector.tick(self.now()):
            self._launch(trigger)

    def ask(self, text: str = "") -> None:
        self._launch(self.detector.manual(text, self.transcript.last("them")))

    # --- approvals -------------------------------------------------------------

    def propose(self, action: ProposedAction) -> Ticket:
        ticket = self.policy.propose(action)
        self.bus.publish("approval", **ticket.to_dict())
        return ticket

    async def decide(self, ticket_id: str, approve: bool, surface: str) -> Ticket:
        ticket = self.policy.decide_ticket(ticket_id, approve, surface)
        if ticket.status == "approved":
            ticket = await self.policy.execute(ticket_id)
        self.bus.publish("approval", **ticket.to_dict())
        return ticket

    def state(self) -> dict:
        """Everything a freshly opened HUD needs to draw itself."""
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "muted": self.gate.muted,
            "title": self.meeting.title if self.meeting else "",
            "brief_items": len(self.brief.items) if self.brief else 0,
            "state": "ended" if self._ended else "listening",
            "suggestions": [{**s.model_dump(), "sources": self._sources.get(s.id, [])} for s in self.suggestions[-8:]],
            "summary": self.summary,
            "transcript": [u.model_dump() for u in self.transcript.finals[-14:]],
            "actions": [a.model_dump() for a in self.actions],
            "approvals": [t.to_dict() for t in self.policy.pending()],
        }

    # --- internals ---------------------------------------------------------------

    def _publish_status(self, state: str, **extra) -> None:
        self.bus.publish("status", state=state, mode=self.mode, muted=self.gate.muted, **extra)

    def _note_commitment(self, trigger: Trigger) -> None:
        commitment, utterance = trigger.commitment, trigger.utterance
        assert commitment is not None and utterance is not None
        text = clip(utterance.text, 200)
        if any(similar(text, a.text) for a in self.actions):
            return
        owner = "я" if commitment.owner == "me" else commitment.owner
        action = ActionItem(id=new_id("act"), text=text, owner=owner, due=commitment.due, utterance_id=utterance.id)
        self.actions.append(action)
        self.bus.publish("action_candidate", **action.model_dump())

    def _cancel_generation(self) -> None:
        if self._generation and not self._generation.done():
            self._generation.cancel()

    def _launch(self, trigger: Trigger) -> None:
        manual = trigger.kind is TriggerKind.MANUAL
        reason = self.gate.preflight(self.now(), manual, trigger.directed)
        if reason:
            self.bus.publish("suppressed", reason=reason, trigger=trigger.kind.value)
            return
        if self._generation and not self._generation.done():
            if self._generation_manual and not manual:
                return
            self._generation.cancel()
        self._generation_manual = manual
        task = asyncio.create_task(self._generate(trigger))
        self._generation = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _focus(self, trigger: Trigger) -> str:
        if trigger.question:
            return trigger.question
        if trigger.utterance:
            return trigger.utterance.text
        last = self.transcript.last("them")
        return last.text if last else ""

    async def _generate(self, trigger: Trigger) -> None:
        suggestion_id = new_id("sg")
        shown = ""
        started = time.monotonic()
        manual = trigger.kind is TriggerKind.MANUAL
        try:
            self._publish_status("thinking", trigger=trigger.kind.value)
            evidence = await self.retriever.retrieve(
                self._focus(trigger), trigger.entities, self.transcript.render(20)
            )
            if evidence.degraded:
                self.bus.publish("warning", message="kernel не ответил вовремя, использую досье перед звонком")
            if not evidence.labels and not trigger.directed:
                self._publish_status("listening")
                return
            request = WhisperRequest(
                user_name=self.settings.user.name or "пользователь",
                language=self.settings.user.language,
                mode=self.mode,
                meeting=self.meeting,
                brief=self.brief,
                evidence=list(evidence.labels.items()),
                transcript=self.transcript.render(),
                trigger_kind=trigger.kind.value,
                trigger_text=self._trigger_text(trigger),
                stall_s=self.settings.live.hesitation_s,
            )
            stream = WhisperStream(set(evidence.labels))
            async with asyncio.timeout(self.settings.llm.timeout_s + 5):
                async for chunk in self.llm.whisper(request):
                    part = stream.feed(chunk)
                    if part and len(part.text) - len(shown) >= PARTIAL_STEP_CHARS:
                        shown = part.text
                        self.bus.publish("suggestion", id=suggestion_id, kind=part.kind, text=part.text, final=False, trigger=trigger.kind.value)
            final = stream.result()
            if final is None or (not trigger.directed and self.gate.is_duplicate(final.text, self.now())):
                if shown:
                    self.bus.publish("suggestion_cancel", id=suggestion_id)
                self._publish_status("listening")
                return
            cited = [evidence.labels[label] for label in final.refs if label in evidence.labels]
            suggestion = Suggestion(
                id=suggestion_id,
                kind=final.kind,
                text=final.text,
                evidence=[item.id for item in cited],
                trigger=trigger.kind.value,
                created_at=self.now(),
            )
            sources = [
                {"id": i.id, "source": i.source, "title": i.title, "ts": i.ts.isoformat() if i.ts else None, "url": i.url}
                for i in cited
            ]
            self.suggestions.append(suggestion)
            self._sources[suggestion.id] = sources
            self.gate.record(final.text, self.now())
            self.bus.publish(
                "suggestion",
                **suggestion.model_dump(),
                sources=sources,
                latency_ms=round((time.monotonic() - started) * 1000),
            )
            self._publish_status("listening", latency_ms=round((time.monotonic() - started) * 1000))
        except asyncio.CancelledError:
            if shown:
                self.bus.publish("suggestion_cancel", id=suggestion_id)
            raise
        except LLMRefused:
            if shown:
                self.bus.publish("suggestion_cancel", id=suggestion_id)
            self._publish_status("listening")
        except Exception as exc:  # a failing model or kernel must never take the call down
            if shown:
                self.bus.publish("suggestion_cancel", id=suggestion_id)
            self._report_error(exc)
            self._publish_status("listening")

    def _trigger_text(self, trigger: Trigger) -> str:
        if trigger.kind is TriggerKind.ENTITY:
            return ", ".join(e.name for e in trigger.entities)
        return self._focus(trigger) if trigger.kind is not TriggerKind.MANUAL else trigger.question

    def _report_error(self, exc: Exception) -> None:
        log.warning("whisper failed: %s: %s", type(exc).__name__, exc)
        now = time.monotonic()
        if now - self._last_error_at >= ERROR_REPORT_GAP_S:
            self._last_error_at = now
            self.bus.publish("error", message=clip(f"{type(exc).__name__}: {exc}", 200))
