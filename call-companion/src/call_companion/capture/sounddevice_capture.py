"""Microphone as "me" and a loopback device (BlackHole) as "them", through PortAudio."""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from .base import AudioFrame, CaptureError, to_mono_16k


class SoundDeviceCapture:
    def __init__(self, mic: str | None, system: str, block_ms: int = 100, sd_module=None) -> None:
        self.mic = mic
        self.system = system
        self.block_ms = block_ms
        self._sd = sd_module

    def _sounddevice(self):
        if self._sd is not None:
            return self._sd
        try:
            import sounddevice
        except ImportError as exc:
            raise CaptureError("не установлен sounddevice: pip install 'call-companion[mac]'") from exc
        return sounddevice

    @staticmethod
    def _resolve(sd, name: str | None, role: str) -> tuple[int | None, int, int]:
        """Return (device index, channels, native sample rate)."""
        if name is None:
            try:
                info = sd.query_devices(kind="input")
            except Exception as exc:
                raise CaptureError(f"нет микрофона по умолчанию ({exc})") from exc
            return None, min(2, int(info["max_input_channels"])) or 1, int(info["default_samplerate"])
        for index, info in enumerate(sd.query_devices()):
            if info["max_input_channels"] > 0 and name.lower() in info["name"].lower():
                return index, min(2, int(info["max_input_channels"])), int(info["default_samplerate"])
        available = ", ".join(i["name"] for i in sd.query_devices() if i["max_input_channels"] > 0)
        hint = " Установите BlackHole 2ch и соберите Multi-Output Device (docs/MACOS_SETUP.md)." if role == "them" else ""
        raise CaptureError(f"не найдено устройство «{name}» для канала {role}. Доступны: {available}.{hint}")

    async def frames(self) -> AsyncIterator[AudioFrame]:
        sd = self._sounddevice()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=400)
        started = time.monotonic()
        opened = []
        closing = False

        def push(frame: AudioFrame) -> None:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(frame)

        try:
            for channel, name in (("me", self.mic), ("them", self.system)):
                index, channels, rate = self._resolve(sd, name, channel)

                def callback(indata, frames, time_info, status, channel=channel, channels=channels, rate=rate):
                    if closing:
                        return
                    pcm = to_mono_16k(bytes(indata), channels, rate)
                    try:
                        loop.call_soon_threadsafe(push, AudioFrame(channel, pcm, time.monotonic() - started))
                    except RuntimeError:  # the loop closed while PortAudio was still delivering; never raise into its thread
                        pass

                stream = sd.RawInputStream(
                    samplerate=rate,
                    channels=channels,
                    dtype="int16",
                    blocksize=rate * self.block_ms // 1000,
                    device=index,
                    callback=callback,
                )
                stream.start()
                opened.append(stream)
            while True:
                yield await queue.get()
        finally:
            closing = True
            for stream in opened:
                stream.stop()
                stream.close()
