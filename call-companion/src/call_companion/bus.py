"""In-process event bus feeding the HUD, the terminal view and tests."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator


@dataclass
class Event:
    id: int
    ts: float
    type: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "ts": self.ts, "type": self.type, "data": self.data}


class EventBus:
    def __init__(self, history: int = 300, queue_size: int = 256) -> None:
        self._seq = 0
        self._history: deque[Event] = deque(maxlen=history)
        self._subscribers: set[asyncio.Queue[Event]] = set()
        self._queue_size = queue_size

    def publish(self, type_: str, **data: Any) -> Event:
        self._seq += 1
        event = Event(self._seq, round(time.time(), 3), type_, data)
        self._history.append(event)
        for queue in list(self._subscribers):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(event)
        return event

    @property
    def last_id(self) -> int:
        return self._seq

    def history(self, after: int = 0) -> list[Event]:
        return [event for event in self._history if event.id > after]

    async def subscribe(self, after: int | None = None, heartbeat_s: float | None = None) -> AsyncIterator[Event | None]:
        """Yield events published from now on, or everything after `after` when it is given.

        With heartbeat_s the generator yields None after that much silence, which lets an SSE
        endpoint send keep-alive pings without cancelling the generator (wait_for on anext would).
        """
        queue: asyncio.Queue[Event] = asyncio.Queue(self._queue_size)
        self._subscribers.add(queue)
        last = self._seq if after is None else after
        try:
            for event in self.history(last):
                last = event.id
                yield event
            while True:
                try:
                    event = await (asyncio.wait_for(queue.get(), heartbeat_s) if heartbeat_s else queue.get())
                except asyncio.TimeoutError:
                    yield None
                    continue
                if event.id > last:
                    last = event.id
                    yield event
        finally:
            self._subscribers.discard(queue)
