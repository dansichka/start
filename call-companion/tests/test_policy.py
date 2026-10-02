import json

import pytest

from call_companion.config import PolicyCfg, PrivacyCfg
from call_companion.models import ContextItem, ProposedAction
from call_companion.policy import AuditLog, PolicyEngine, Redactor


def action(capability: str, **payload) -> ProposedAction:
    return ProposedAction(id="a1", capability=capability, title=f"do {capability}", payload=payload)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_tiers():
    engine = PolicyEngine()
    assert engine.decide("kernel.read").allowed
    assert not engine.decide("kernel.read").needs_approval
    assert not engine.decide("email.draft").needs_approval
    assert engine.decide("kernel.note").needs_approval
    assert engine.decide("email.send").needs_approval
    assert not engine.decide("shell.exec").allowed
    assert not engine.decide("something.unknown").allowed


def test_auto_allow_covers_notes_but_never_sends():
    engine = PolicyEngine(PolicyCfg(auto_allow=["kernel.note", "email.send"]))
    assert not engine.decide("kernel.note").needs_approval
    assert engine.decide("email.send").needs_approval


def test_propose_sets_initial_status():
    engine = PolicyEngine()
    assert engine.propose(action("kernel.read")).status == "approved"
    assert engine.propose(action("kernel.note")).status == "pending"
    assert engine.propose(action("email.send")).status == "pending"
    assert engine.propose(action("shell.exec")).status == "rejected"


def test_only_local_surfaces_can_approve():
    engine = PolicyEngine()
    ticket = engine.propose(action("email.send", to="x@y.z"))
    for surface in ("voice", "llm", "them", "mic"):
        with pytest.raises(PermissionError):
            engine.decide_ticket(ticket.id, True, surface)
    assert engine.ticket(ticket.id).status == "pending"
    assert engine.decide_ticket(ticket.id, True, "hud").status == "approved"
    with pytest.raises(ValueError):
        engine.decide_ticket(ticket.id, False, "hud")


async def test_execute_runs_once_and_only_after_approval():
    engine = PolicyEngine()
    calls = []

    async def send(act: ProposedAction) -> str:
        calls.append(act.id)
        return "sent"

    engine.register_executor("email.send", send)
    ticket = engine.propose(action("email.send"))
    with pytest.raises(ValueError):
        await engine.execute(ticket.id)
    engine.decide_ticket(ticket.id, True, "hotkey")
    done = await engine.execute(ticket.id)
    assert done.status == "executed" and done.result == "sent" and calls == ["a1"]
    with pytest.raises(ValueError):
        await engine.execute(ticket.id)


async def test_executor_failure_is_contained():
    engine = PolicyEngine(PolicyCfg(auto_allow=["kernel.note"]))

    async def boom(act: ProposedAction) -> str:
        raise RuntimeError("kernel offline")

    engine.register_executor("kernel.note", boom)
    ticket = engine.propose(action("kernel.note"))
    done = await engine.execute(ticket.id)
    assert done.status == "failed" and "kernel offline" in done.result


async def test_missing_executor_marks_failed():
    engine = PolicyEngine(PolicyCfg(auto_allow=["kernel.note"]))
    ticket = engine.propose(action("kernel.note"))
    assert (await engine.execute(ticket.id)).status == "failed"


def test_stale_approvals_expire():
    clock = Clock()
    engine = PolicyEngine(PolicyCfg(approval_ttl_s=60), clock=clock)
    ticket = engine.propose(action("email.send"))
    clock.now += 61
    assert engine.ticket(ticket.id).status == "expired"
    assert engine.pending() == []
    with pytest.raises(ValueError):
        engine.decide_ticket(ticket.id, True, "hud")


def test_redaction_of_secrets():
    redact = Redactor()
    assert "sk-ant" not in redact("ключ sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123")
    assert "[скрыто]" in redact("пароль: hunter2")
    assert "hunter2" not in redact("password = hunter2")
    assert "ghp_" not in redact("tok ghp_" + "a" * 36)
    assert "123456789:" not in redact("bot 123456789:" + "A" * 35)
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r"
    assert "eyJ" not in redact(jwt)


def test_redaction_validates_cards_and_ibans():
    redact = Redactor()
    assert "[скрыто:карта]" in redact("карта 4111 1111 1111 1111 ок")
    assert "1234 5678 9012 3456" in redact("номер 1234 5678 9012 3456")
    assert "[скрыто:iban]" in redact("счёт GB82 WEST 1234 5698 7654 32")
    assert "DE00 1234 5678 9012 3456 78" in redact("DE00 1234 5678 9012 3456 78")
    assert redact("+380 50 123 45 67") == "+380 50 123 45 67"


def test_extra_patterns_and_disabled_redaction():
    assert "[скрыто]" in PolicyEngine(privacy=PrivacyCfg(extra_redact_patterns=[r"PROJ-\d+"])).redact("see PROJ-42")
    assert PolicyEngine(privacy=PrivacyCfg(redact=False)).redact("пароль: x") == "пароль: x"


def make_items() -> list[ContextItem]:
    return [
        ContextItem(id="pub", source="doc", sensitivity="public", text="открыто"),
        ContextItem(id="wrk", source="note", sensitivity="work", text="рабочее, token: abc123"),
        ContextItem(id="prv", source="telegram", sensitivity="private", text="личное"),
        ContextItem(id="sec", source="stripe", sensitivity="secret", text="финансы"),
    ]


def test_filter_items_by_mode():
    engine = PolicyEngine()
    assert [i.id for i in engine.filter_items(make_items(), "client")] == ["pub", "wrk"]
    assert [i.id for i in engine.filter_items(make_items(), "internal")] == ["pub", "wrk", "prv"]
    assert [i.id for i in engine.filter_items(make_items(), "public")] == ["pub"]
    assert "abc123" not in engine.filter_items(make_items(), "client")[1].text


def test_mode_limits_can_be_overridden():
    engine = PolicyEngine(PolicyCfg(mode_limits={"client": "private"}))
    assert "prv" in [i.id for i in engine.filter_items(make_items(), "client")]
    assert "sec" not in [i.id for i in engine.filter_items(make_items(), "internal")]


def test_audit_chain_detects_tampering(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for event in ("a", "b", "c"):
        log.append(event, n=1)
    assert log.verify() == (True, None)
    assert oct(path.stat().st_mode & 0o777) == "0o600"

    lines = path.read_text(encoding="utf-8").splitlines()
    forged = json.loads(lines[1])
    forged["event"] = "forged"
    lines[1] = json.dumps(forged, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert AuditLog(path).verify() == (False, 1)


def test_audit_chain_continues_across_restarts(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLog(path).append("first")
    second = AuditLog(path)
    second.append("second")
    assert second.verify() == (True, None)


def test_audit_never_stores_payload_contents(tmp_path):
    engine = PolicyEngine(audit=AuditLog(tmp_path / "a.jsonl"))
    engine.propose(action("email.send", body="СЕКРЕТНЫЙ ТЕКСТ ПИСЬМА"))
    assert "СЕКРЕТНЫЙ" not in (tmp_path / "a.jsonl").read_text(encoding="utf-8")
