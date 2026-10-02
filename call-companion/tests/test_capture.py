import asyncio
import stat
import struct
import sys
import textwrap
import threading
import time
import types

import numpy as np
import pytest

from call_companion.capture import build_capture
from call_companion.capture.audiotap import AudioTapCapture
from call_companion.capture.base import CaptureError, to_mono_16k
from call_companion.capture.sounddevice_capture import SoundDeviceCapture
from call_companion.config import CaptureCfg


def sine(seconds: float, rate: int, freq: float = 440.0, channels: int = 1) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    wave = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    return (np.repeat(wave[:, None], channels, axis=1) if channels > 1 else wave).tobytes()


def crossings(pcm: bytes) -> int:
    x = np.frombuffer(pcm, dtype=np.int16)
    return int(np.sum(np.diff(np.sign(x)) != 0))


def test_conversion_downmixes_and_resamples_to_16k_mono():
    out = to_mono_16k(sine(1.0, 48000, channels=2), channels=2, rate=48000)
    assert len(out) // 2 == 16000
    assert crossings(out) == pytest.approx(880, abs=6), "a 440 Hz tone keeps its pitch"
    odd = to_mono_16k(sine(1.0, 44100), channels=1, rate=44100)
    assert len(odd) // 2 == 16000 and crossings(odd) == pytest.approx(880, abs=8)
    assert to_mono_16k(sine(0.1, 16000), 1, 16000) == sine(0.1, 16000)


def test_conversion_tolerates_empty_and_ragged_input():
    assert to_mono_16k(b"", 2, 48000) == b""
    assert len(to_mono_16k(sine(0.1, 48000, channels=2) + b"\x01", channels=2, rate=48000)) == 1600 * 2


# --- sounddevice with a fake PortAudio ------------------------------------------------


class FakeStream:
    instances: list = []

    def __init__(self, samplerate, channels, dtype, blocksize, device, callback):
        self.rate, self.channels, self.device, self.callback, self.blocksize = samplerate, channels, device, callback, blocksize
        self.stopped = self.closed = False
        FakeStream.instances.append(self)

    def start(self):
        def feed():
            for _ in range(3):
                if self.stopped:
                    return
                self.callback(sine(0.1, self.rate, channels=self.channels), self.blocksize, None, None)
                time.sleep(0.01)

        threading.Thread(target=feed, daemon=True).start()

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def fake_sd(devices: list[dict], default_input: dict | None = None) -> types.SimpleNamespace:
    FakeStream.instances.clear()

    def query_devices(kind=None):
        if kind == "input":
            if default_input is None:
                raise RuntimeError("no default input")
            return default_input
        return devices

    return types.SimpleNamespace(query_devices=query_devices, RawInputStream=FakeStream)


DEVICES = [
    {"name": "MacBook Air Microphone", "max_input_channels": 1, "default_samplerate": 48000.0},
    {"name": "BlackHole 2ch", "max_input_channels": 2, "default_samplerate": 48000.0},
    {"name": "MacBook Air Speakers", "max_input_channels": 0, "default_samplerate": 48000.0},
]


async def test_sounddevice_labels_channels_and_converts_to_16k_mono():
    capture = SoundDeviceCapture(None, "blackhole", sd_module=fake_sd(DEVICES, DEVICES[0]))
    frames, stream = [], capture.frames()
    async with asyncio.timeout(3):
        async for frame in stream:
            frames.append(frame)
            if len(frames) == 6:
                break
    await stream.aclose()
    assert {f.channel for f in frames} == {"me", "them"}
    assert all(len(f.pcm) == 1600 * 2 for f in frames)
    assert all(crossings(f.pcm) > 60 for f in frames)
    them = sorted(FakeStream.instances, key=lambda s: s.channels)[-1]
    assert them.device == 1 and them.channels == 2 and them.blocksize == 4800
    assert all(s.stopped and s.closed for s in FakeStream.instances)


async def test_a_named_microphone_is_matched_by_substring():
    capture = SoundDeviceCapture("air microphone", "BlackHole", sd_module=fake_sd(DEVICES))
    stream = capture.frames()
    async with asyncio.timeout(3):
        await anext(stream)
    await stream.aclose()
    assert sorted(s.device for s in FakeStream.instances) == [0, 1]


async def test_missing_loopback_explains_how_to_fix_it():
    capture = SoundDeviceCapture(None, "BlackHole 2ch", sd_module=fake_sd(DEVICES[:1], DEVICES[0]))
    with pytest.raises(CaptureError, match="BlackHole") as error:
        await anext(capture.frames())
    assert "MACOS_SETUP" in str(error.value) and "MacBook Air Microphone" in str(error.value)


async def test_no_default_microphone_is_reported():
    with pytest.raises(CaptureError, match="нет микрофона"):
        await anext(SoundDeviceCapture(None, "BlackHole", sd_module=fake_sd(DEVICES)).frames())


async def test_missing_sounddevice_gives_an_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(CaptureError, match=r"call-companion\[mac\]"):
        await anext(SoundDeviceCapture(None, "BlackHole").frames())


# --- the native helper's wire protocol ---------------------------------------------------------


def helper(tmp_path, body: str):
    path = tmp_path / "audiotap"
    path.write_text(f"#!{sys.executable}\nimport struct, sys\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


async def collect(capture) -> list:
    return [f async for f in capture.frames()]


async def test_audiotap_demultiplexes_both_channels(tmp_path):
    binary = helper(
        tmp_path,
        """
        pcm = struct.pack("<4h", 1, 2, 3, 4)
        for channel in (0, 1, 1, 0):
            sys.stdout.buffer.write(struct.pack("<BI", channel, len(pcm)) + pcm)
        sys.stdout.buffer.flush()
        """,
    )
    frames = await collect(AudioTapCapture(binary))
    assert [f.channel for f in frames] == ["me", "them", "them", "me"]
    assert all(f.pcm == struct.pack("<4h", 1, 2, 3, 4) for f in frames)
    assert [f.t for f in frames] == sorted(f.t for f in frames)


async def test_audiotap_failure_carries_the_helpers_message(tmp_path):
    binary = helper(tmp_path, 'sys.stderr.write("нет доступа к записи экрана и системного звука"); sys.exit(3)')
    with pytest.raises(CaptureError, match="кодом 3.*записи экрана"):
        await collect(AudioTapCapture(binary))


async def test_audiotap_rejects_a_corrupt_stream(tmp_path):
    binary = helper(tmp_path, 'sys.stdout.buffer.write(struct.pack("<BI", 7, 4)); sys.stdout.buffer.flush()')
    with pytest.raises(CaptureError, match="повреждённый"):
        await collect(AudioTapCapture(binary))
    huge = helper(tmp_path, 'sys.stdout.buffer.write(struct.pack("<BI", 0, 2**31)); sys.stdout.buffer.flush()')
    with pytest.raises(CaptureError, match="повреждённый"):
        await collect(AudioTapCapture(huge))


async def test_audiotap_is_stopped_when_the_consumer_leaves(tmp_path):
    binary = helper(
        tmp_path,
        """
        import time
        pcm = struct.pack("<2h", 1, 2)
        while True:
            sys.stdout.buffer.write(struct.pack("<BI", 0, len(pcm)) + pcm); sys.stdout.buffer.flush(); time.sleep(0.02)
        """,
    )
    stream = AudioTapCapture(binary).frames()
    assert (await anext(stream)).channel == "me"
    await asyncio.wait_for(stream.aclose(), 5)


async def test_missing_binary_points_to_the_build_instructions(tmp_path):
    with pytest.raises(CaptureError, match="MACOS_SETUP"):
        await collect(AudioTapCapture(tmp_path / "nope"))


def test_build_capture_picks_the_backend(tmp_path):
    assert isinstance(build_capture(CaptureCfg(backend="audiotap", audiotap_path=str(tmp_path / "x"))), AudioTapCapture)
    assert isinstance(build_capture(CaptureCfg(backend="sounddevice")), SoundDeviceCapture)


async def test_a_callback_after_shutdown_never_raises_into_the_audio_thread():
    sd = fake_sd(DEVICES, DEVICES[0])
    stream = SoundDeviceCapture(None, "BlackHole", sd_module=sd).frames()
    async with asyncio.timeout(3):
        await anext(stream)
    await stream.aclose()
    for opened in FakeStream.instances:
        opened.callback(sine(0.1, opened.rate, channels=opened.channels), opened.blocksize, None, None)


def test_a_closed_event_loop_is_tolerated_by_the_callback():
    sd = fake_sd(DEVICES, DEVICES[0])

    async def open_and_leave():
        stream = SoundDeviceCapture(None, "BlackHole", sd_module=sd).frames()
        await anext(stream)
        return stream

    loop = asyncio.new_event_loop()
    stream = loop.run_until_complete(open_and_leave())
    loop.close()
    for opened in FakeStream.instances:
        opened.callback(sine(0.1, opened.rate, channels=opened.channels), opened.blocksize, None, None)
    del stream
