"""Local session storage; raw transcripts are kept only when the user opted in."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from ..models import CallRecord, CallSummary


class SessionStore:
    def __init__(self, data_dir: Path, *, keep_transcripts: bool = False, retention_days: int = 14) -> None:
        self.root = data_dir / "sessions"
        self.keep_transcripts = keep_transcripts
        self.retention_days = retention_days

    def dir_for(self, session_id: str) -> Path:
        path = self.root / session_id
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o700)
        return path

    @staticmethod
    def _write(path: Path, content: str) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)

    def save(self, record: CallRecord, summary: CallSummary | None) -> Path:
        path = self.dir_for(record.session_id)
        meta = {
            "session_id": record.session_id,
            "title": record.meeting.title,
            "started_at": record.started_at.isoformat(),
            "ended_at": record.ended_at.isoformat() if record.ended_at else None,
            "utterances": len(record.utterances),
            "suggestions": len(record.suggestions),
            "transcript_stored": self.keep_transcripts,
        }
        self._write(path / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
        self._write(
            path / "actions.json",
            json.dumps([a.model_dump() for a in record.action_items], ensure_ascii=False, indent=2),
        )
        if summary:
            self._write(path / "summary.json", summary.model_dump_json(indent=2))
        if self.keep_transcripts:
            rows = (u.model_dump_json() for u in record.utterances)
            self._write(path / "transcript.jsonl", "\n".join(rows) + "\n")
            self._write(path / "suggestions.jsonl", "\n".join(s.model_dump_json() for s in record.suggestions) + "\n")
        return path

    def purge(self, now: float | None = None) -> int:
        """Delete sessions older than the retention window; returns how many were removed."""
        if not self.root.exists():
            return 0
        horizon = (now or time.time()) - self.retention_days * 86400
        removed = 0
        for path in self.root.iterdir():
            if path.is_dir() and path.stat().st_mtime < horizon:
                shutil.rmtree(path)
                removed += 1
        return removed
