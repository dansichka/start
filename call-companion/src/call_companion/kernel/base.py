"""Client side of the kernel contract (see docs/KERNEL_CONTRACT.md)."""

from __future__ import annotations

from typing import Protocol

from ..models import Brief, ContextItem, Health, Meeting, Note, ProposedAction, SearchFilters


class KernelError(Exception):
    pass


class KernelUnavailable(KernelError):
    """Kernel cannot be reached right now; callers fall back to local data."""


class KernelAuthError(KernelError):
    pass


class KernelClient(Protocol):
    name: str

    async def health(self) -> Health: ...

    async def search(self, query: str, *, k: int = 6, filters: SearchFilters | None = None) -> list[ContextItem]: ...

    async def brief(self, meeting: Meeting, *, max_sensitivity: str = "work") -> Brief: ...

    async def upcoming(self, within_min: int = 30) -> list[Meeting]: ...

    async def save_note(self, note: Note) -> str: ...

    async def submit_action(self, action: ProposedAction) -> str: ...

    async def aclose(self) -> None: ...


class BaseKernel:
    """Defaults for backends that only implement part of the contract."""

    name = "base"

    async def upcoming(self, within_min: int = 30) -> list[Meeting]:
        return []

    async def submit_action(self, action: ProposedAction) -> str:
        raise KernelUnavailable(f"{self.name}: действия не поддерживаются")

    async def aclose(self) -> None:
        return None
