"""Frames from the native helper (native/macos/AudioTap): ScreenCaptureKit for system audio, AVAudioEngine for the mic.

Wire format on the helper's stdout, repeated: 1 byte channel (0 = me, 1 = them), uint32 little-endian byte
length, then that many bytes of int16 mono PCM at 16 kHz. Diagnostics go to stderr; exit code 0 is a clean stop.
"""

from __future__ import annotations

import asyncio
import signal
import struct
import time
from pathlib import Path
from typing import AsyncIterator

from .base import AudioFrame, CaptureError

HEADER = struct.Struct("<BI")
MAX_FRAME_BYTES = 1_000_000


class AudioTapCapture:
    def __init__(self, binary: str | Path) -> None:
        self.binary = Path(binary).expanduser()

    async def frames(self) -> AsyncIterator[AudioFrame]:
        if not self.binary.exists():
            raise CaptureError(f"не найден {self.binary}: соберите native/macos/AudioTap (docs/MACOS_SETUP.md)")
        process = await asyncio.create_subprocess_exec(
            str(self.binary), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        started = time.monotonic()
        try:
            while True:
                try:
                    channel, length = HEADER.unpack(await process.stdout.readexactly(HEADER.size))
                    if channel not in (0, 1) or length > MAX_FRAME_BYTES:
                        raise CaptureError("audiotap прислал повреждённый кадр")
                    pcm = await process.stdout.readexactly(length)
                except asyncio.IncompleteReadError:
                    break
                yield AudioFrame("me" if channel == 0 else "them", pcm, time.monotonic() - started)
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 2)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
        if process.returncode not in (0, -signal.SIGTERM):
            tail = (await process.stderr.read()).decode("utf-8", errors="replace")[-300:].strip()
            raise CaptureError(f"audiotap завершился с кодом {process.returncode}: {tail}")
