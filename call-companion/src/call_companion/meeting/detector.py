"""Pick the meeting to attach to: the calendar decides, running call apps are a supporting signal."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..models import Meeting

CALL_APPS = {
    "zoom.us": "zoom",
    "microsoft teams": "teams",
    "msteams": "teams",
    "facetime": "facetime",
    "webex": "webex",
    "skype": "skype",
    "discord": "discord",
    "slack": "slack",
    "telegram": "telegram",
    "whatsapp": "whatsapp",
}


def running_call_apps(process_names: list[str] | None = None) -> list[str]:
    """Names of call apps among the running processes (a browser tab with Meet cannot be seen this way)."""
    if process_names is None:
        import psutil

        process_names = [p.info["name"] or "" for p in psutil.process_iter(["name"])]
    found = {label for name in process_names for key, label in CALL_APPS.items() if key in name.lower()}
    return sorted(found)


def pick_meeting(
    meetings: list[Meeting],
    now: datetime | None = None,
    *,
    early: timedelta = timedelta(minutes=5),
    late: timedelta = timedelta(minutes=20),
) -> Meeting | None:
    """The running meeting that started most recently, or the next one starting within `early`; finished ones are skipped."""
    now = now or datetime.now(timezone.utc)
    candidates = [
        m for m in meetings if m.start and now - late <= m.start <= now + early and not (m.end and m.end < now)
    ]
    if not candidates:
        return None
    started = [m for m in candidates if m.start <= now]
    return max(started, key=lambda m: m.start) if started else min(candidates, key=lambda m: m.start)
