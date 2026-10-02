"""Access control: capability tiers, approvals, redaction and a tamper-evident audit trail.

Audio from other participants is untrusted input. Nothing said on a call can trigger an
action above the READ tier without a human decision taken on a local surface (HUD click,
hotkey or CLI) - never by voice, because a loudspeaker can replay anyone's voice into the mic.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Awaitable, Callable, Literal

from .config import PolicyCfg, PrivacyCfg
from .models import ContextItem, MeetingMode, ProposedAction, sensitivity_rank
from .redact import Redactor
from .util import canonical_json, new_id


class Risk(IntEnum):
    READ = 0
    NOTE = 1
    DRAFT = 2
    SEND = 3
    PRIVILEGED = 4


CAPABILITIES: dict[str, Risk] = {
    "kernel.read": Risk.READ,
    "kernel.note": Risk.NOTE,
    "kernel.task": Risk.NOTE,
    "email.draft": Risk.DRAFT,
    "message.draft": Risk.DRAFT,
    "email.send": Risk.SEND,
    "message.send": Risk.SEND,
    "calendar.write": Risk.SEND,
    "shell.exec": Risk.PRIVILEGED,
}

MODE_LIMITS: dict[str, str] = {
    "internal": "private",
    "client": "work",
    "interview": "work",
    "public": "public",
}

TRUSTED_SURFACES = frozenset({"hud", "hotkey", "cli"})
Status = Literal["pending", "approved", "rejected", "expired", "executed", "failed"]
Executor = Callable[[ProposedAction], Awaitable[str]]


@dataclass(frozen=True)
class Decision:
    allowed: bool
    needs_approval: bool
    reason: str


@dataclass
class Ticket:
    id: str
    action: ProposedAction
    status: Status
    created_at: float
    expires_at: float
    decided_by: str | None = None
    result: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["action"] = self.action.model_dump(mode="json")
        return data


class AuditLog:
    """Append-only JSONL where every record carries the hash of the previous one."""

    GENESIS = "0" * 64

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._memory: list[dict] = []
        self._last = self.GENESIS
        if path and path.exists():
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if lines:
                self._last = json.loads(lines[-1])["hash"]

    def append(self, event: str, **fields: object) -> dict:
        record: dict = {"ts": round(time.time(), 3), "event": event, **fields, "prev": self._last}
        with self._lock:
            record["prev"] = self._last
            record["hash"] = hashlib.sha256((self._last + canonical_json(record)).encode()).hexdigest()
            self._last = record["hash"]
            self._memory.append(record)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
                with os.fdopen(fd, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def verify(self) -> tuple[bool, int | None]:
        """Return (ok, index of the first broken record)."""
        records = self._memory
        if self.path and self.path.exists():
            records = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        prev = self.GENESIS
        for index, record in enumerate(records):
            claimed = record.get("hash")
            body = {k: v for k, v in record.items() if k != "hash"}
            expected = hashlib.sha256((prev + canonical_json(body)).encode()).hexdigest()
            if body.get("prev") != prev or claimed != expected:
                return False, index
            prev = claimed
        return True, None


class PolicyEngine:
    def __init__(
        self,
        policy: PolicyCfg | None = None,
        privacy: PrivacyCfg | None = None,
        audit: AuditLog | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.policy = policy or PolicyCfg()
        self.privacy = privacy or PrivacyCfg()
        self.audit = audit or AuditLog()
        self._clock = clock
        self._redactor = Redactor(self.privacy.extra_redact_patterns)
        self._tickets: dict[str, Ticket] = {}
        self._executors: dict[str, Executor] = {}

    def redact(self, text: str) -> str:
        return self._redactor(text) if self.privacy.redact else text

    def sensitivity_limit(self, mode: MeetingMode) -> str:
        return self.policy.mode_limits.get(mode) or MODE_LIMITS.get(mode, "work")

    def filter_items(self, items: list[ContextItem], mode: MeetingMode) -> list[ContextItem]:
        limit = sensitivity_rank(self.sensitivity_limit(mode))
        kept: list[ContextItem] = []
        for item in items:
            rank = sensitivity_rank(item.sensitivity)
            if item.sensitivity == "secret" or rank > limit:
                continue
            kept.append(item.model_copy(update={"title": self.redact(item.title), "text": self.redact(item.text)}))
        if len(kept) != len(items):
            self.audit.append("filter", mode=mode, kept=len(kept), dropped=len(items) - len(kept))
        return kept

    def decide(self, capability: str) -> Decision:
        risk = CAPABILITIES.get(capability, Risk.PRIVILEGED)
        if risk >= Risk.PRIVILEGED:
            return Decision(False, False, "недоступно: привилегированные действия отключены")
        if risk >= Risk.SEND:
            return Decision(True, True, "внешний эффект: каждое действие подтверждается вручную")
        if risk == Risk.READ or risk == Risk.DRAFT:
            return Decision(True, False, "локальное действие без внешнего эффекта")
        if capability in self.policy.auto_allow:
            return Decision(True, False, "разрешено в конфиге")
        return Decision(True, True, "запись в kernel требует подтверждения")

    def register_executor(self, capability: str, executor: Executor) -> None:
        self._executors[capability] = executor

    def propose(self, action: ProposedAction) -> Ticket:
        decision = self.decide(action.capability)
        now = self._clock()
        status: Status = "rejected" if not decision.allowed else ("pending" if decision.needs_approval else "approved")
        ticket = Ticket(
            id=new_id("tk"),
            action=action,
            status=status,
            created_at=now,
            expires_at=now + self.policy.approval_ttl_s,
            decided_by=None if decision.needs_approval or not decision.allowed else "policy",
        )
        self._tickets[ticket.id] = ticket
        self.audit.append(
            "propose",
            ticket=ticket.id,
            capability=action.capability,
            origin=action.origin,
            title=action.title[:120],
            payload=hashlib.sha256(canonical_json(action.payload).encode()).hexdigest()[:16],
            status=status,
            reason=decision.reason,
        )
        return ticket

    def pending(self) -> list[Ticket]:
        self._expire()
        return [t for t in self._tickets.values() if t.status == "pending"]

    def ticket(self, ticket_id: str) -> Ticket | None:
        self._expire()
        return self._tickets.get(ticket_id)

    def decide_ticket(self, ticket_id: str, approve: bool, surface: str) -> Ticket:
        ticket = self.ticket(ticket_id)
        if ticket is None:
            raise KeyError(ticket_id)
        if surface not in TRUSTED_SURFACES:
            self.audit.append("decision_refused", ticket=ticket_id, surface=surface)
            raise PermissionError(f"поверхность {surface!r} не может подтверждать действия")
        if ticket.status != "pending":
            raise ValueError(f"тикет в статусе {ticket.status}")
        ticket.status = "approved" if approve else "rejected"
        ticket.decided_by = surface
        self.audit.append("decision", ticket=ticket_id, approved=approve, surface=surface)
        return ticket

    async def execute(self, ticket_id: str) -> Ticket:
        ticket = self.ticket(ticket_id)
        if ticket is None:
            raise KeyError(ticket_id)
        if ticket.status != "approved":
            raise ValueError(f"выполнять можно только одобренное, сейчас: {ticket.status}")
        executor = self._executors.get(ticket.action.capability)
        if executor is None:
            ticket.status, ticket.result = "failed", "нет исполнителя для этого действия"
        else:
            try:
                ticket.result = await executor(ticket.action)
                ticket.status = "executed"
            except Exception as exc:  # executor failures must not break the call
                ticket.status, ticket.result = "failed", f"{type(exc).__name__}: {exc}"
        self.audit.append("execute", ticket=ticket_id, status=ticket.status)
        return ticket

    def _expire(self) -> None:
        now = self._clock()
        for ticket in self._tickets.values():
            if ticket.status in ("pending", "approved") and now > ticket.expires_at:
                ticket.status = "expired"
                self.audit.append("expire", ticket=ticket.id)
