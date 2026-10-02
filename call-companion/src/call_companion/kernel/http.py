"""HTTP transport to the kernel host (LAN or Tailscale), with a short budget and a circuit breaker."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import httpx
from pydantic import ValidationError

from ..models import Brief, ContextItem, Health, Meeting, Note, ProposedAction, SearchFilters
from ..util import canonical_json
from .base import BaseKernel, KernelAuthError, KernelError, KernelUnavailable

log = logging.getLogger(__name__)


class HttpKernel(BaseKernel):
    name = "http"

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        *,
        timeout_s: float = 0.8,
        brief_timeout_s: float = 8.0,
        cache_ttl_s: float = 60.0,
        breaker_threshold: int = 3,
        breaker_reset_s: float = 20.0,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_s, transport=transport
        )
        self._timeout_s = timeout_s
        self._brief_timeout_s = brief_timeout_s
        self._cache_ttl_s = cache_ttl_s
        self._threshold = breaker_threshold
        self._reset_s = breaker_reset_s
        self._clock = clock
        self._failures = 0
        self._open_until = 0.0
        self._cache: dict[str, tuple[float, list[ContextItem]]] = {}

    async def _request(
        self, method: str, path: str, *, json: Any = None, params: Any = None, timeout: float | None = None
    ) -> dict[str, Any]:
        now = self._clock()
        if now < self._open_until:
            raise KernelUnavailable("kernel http: цепь разомкнута, пропускаю запрос")
        try:
            response = await self._client.request(method, path, json=json, params=params, timeout=timeout or self._timeout_s)
        except httpx.TransportError as exc:
            self._record_failure()
            raise KernelUnavailable(f"kernel http: {type(exc).__name__}") from exc
        if response.status_code in (401, 403):
            raise KernelAuthError(f"kernel http {response.status_code}: проверьте токен")
        if response.status_code >= 500:
            self._record_failure()
            raise KernelUnavailable(f"kernel http {response.status_code}")
        if response.status_code >= 400:
            raise KernelError(f"kernel http {response.status_code}: {response.text[:200]}")
        self._failures = 0
        try:
            return response.json()
        except ValueError as exc:
            raise KernelError("kernel http: ответ не JSON") from exc

    def _record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._open_until = self._clock() + self._reset_s
            self._failures = 0
            log.warning("kernel http: %d сбоев подряд, пауза %.0f с", self._threshold, self._reset_s)

    async def health(self) -> Health:
        started = self._clock()
        data = await self._request("GET", "/v1/health")
        return Health(**{**data, "latency_ms": round((self._clock() - started) * 1000, 1)})

    async def search(self, query: str, *, k: int = 6, filters: SearchFilters | None = None) -> list[ContextItem]:
        filters = filters or SearchFilters()
        body = {"query": query, "k": k, "filters": filters.model_dump(mode="json")}
        key = canonical_json(body)
        cached = self._cache.get(key)
        if cached and cached[0] > self._clock():
            return cached[1]
        data = await self._request("POST", "/v1/search", json=body)
        items = self._parse_items(data.get("items", []))
        if len(self._cache) >= 128:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (self._clock() + self._cache_ttl_s, items)
        return items

    @staticmethod
    def _parse_items(rows: list[dict[str, Any]]) -> list[ContextItem]:
        items = []
        for row in rows:
            try:
                items.append(ContextItem.model_validate(row))
            except ValidationError as exc:
                log.warning("kernel http: элемент пропущен: %s", exc.errors()[0]["msg"])
        return items

    async def brief(self, meeting: Meeting, *, max_sensitivity: str = "work") -> Brief:
        body = {"meeting": meeting.model_dump(mode="json"), "max_sensitivity": max_sensitivity}
        data = await self._request("POST", "/v1/brief", json=body, timeout=self._brief_timeout_s)
        try:
            return Brief.model_validate(data)
        except ValidationError as exc:
            raise KernelError(f"kernel http: некорректный brief: {exc.errors()[0]['msg']}") from exc

    async def upcoming(self, within_min: int = 30) -> list[Meeting]:
        data = await self._request("GET", "/v1/meetings/upcoming", params={"within_min": within_min})
        return [Meeting.model_validate(row) for row in data.get("meetings", [])]

    async def save_note(self, note: Note) -> str:
        data = await self._request("POST", "/v1/notes", json=note.model_dump(mode="json"), timeout=self._brief_timeout_s)
        return str(data["id"])

    async def submit_action(self, action: ProposedAction) -> str:
        data = await self._request("POST", "/v1/actions", json=action.model_dump(mode="json"), timeout=self._brief_timeout_s)
        return str(data["id"])

    async def aclose(self) -> None:
        await self._client.aclose()
