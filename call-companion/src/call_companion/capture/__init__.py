"""Audio sources for live calls."""

from .base import AudioFrame, AudioSource, CaptureError

__all__ = ["AudioFrame", "AudioSource", "CaptureError", "build_capture"]


def build_capture(cfg) -> AudioSource:
    if cfg.backend == "audiotap":
        from .audiotap import AudioTapCapture

        return AudioTapCapture(cfg.audiotap_path)
    from .sounddevice_capture import SoundDeviceCapture

    return SoundDeviceCapture(cfg.mic_device, cfg.system_device)
