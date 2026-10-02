"""CLI transport: runs `<command> <op>` with a JSON request on stdin and reads JSON from stdout.

The command can be a local binary or a prefix such as ["ssh", "pc", "kernel", "companion"].
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Sequence

from pydantic import ValidationError

from ..models import Brief, ContextItem, Health, Meeting, Note, ProposedAction, SearchFilters
from .base import BaseKernel, KernelError, KernelUnavailable


class CliKernel(BaseKernel):
    name = "cli"

    def __init__(self, command: Sequence[str], timeout_s: float = 6.0) -> None:
        self.command = list(command)
        self.timeout_s = timeout_s

    async def _call(self, op: str, payload: dict[str, Any] | None = None, timeout: float | None = None) -> dict[str, Any]:
        try:
            process = await asyncio.create_subprocess_exec(
                *self.command,
                op,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, PermissionError) as exc:
            raise KernelUnavailable(f"kernel cli: не удалось запустить {self.command[0]}: {exc}") from exc
        request = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(request), timeout or self.timeout_s)
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            raise KernelUnavailable(f"kernel cli: {op} не уложился в {timeout or self.timeout_s:.1f} с") from exc
        if process.returncode != 0:
            tail = stderr.decode("utf-8", errors="replace")[-300:].strip()
            raise KernelError(f"kernel cli: {op} завершился с кодом {process.returncode}: {tail}")
        try:
            return json.loads(stdout.decode("utf-8"))
        except ValueError as exc:
            raise KernelError(f"kernel cli: {op} вернул не JSON") from exc

    async def health(self) -> Health:
        return Health.model_validate(await self._call("health"))

    async def search(self, query: str, *, k: int = 6, filters: SearchFilters | None = None) -> list[ContextItem]:
        body = {"query": query, "k": k, "filters": (filters or SearchFilters()).model_dump(mode="json")}
        data = await self._call("search", body)
        try:
            return [ContextItem.model_validate(row) for row in data.get("items", [])]
        except ValidationError as exc:
            raise KernelError(f"kernel cli: некорректный элемент: {exc.errors()[0]['msg']}") from exc

    async def brief(self, meeting: Meeting, *, max_sensitivity: str = "work") -> Brief:
        body = {"meeting": meeting.model_dump(mode="json"), "max_sensitivity": max_sensitivity}
        return Brief.model_validate(await self._call("brief", body, timeout=self.timeout_s * 3))

    async def upcoming(self, within_min: int = 30) -> list[Meeting]:
        data = await self._call("upcoming", {"within_min": within_min})
        return [Meeting.model_validate(row) for row in data.get("meetings", [])]

    async def save_note(self, note: Note) -> str:
        return str((await self._call("note", note.model_dump(mode="json")))["id"])

    async def submit_action(self, action: ProposedAction) -> str:
        return str((await self._call("action", action.model_dump(mode="json")))["id"])
