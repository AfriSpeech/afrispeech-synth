"""TTS backend interface and the audio plumbing every backend shares."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional


class TTSError(Exception):
    pass


class RetryableTTSError(TTSError):
    """Rate limit or transient server error — worth backing off and retrying."""


@dataclass
class Clip:
    audio: bytes
    mime_type: str
    sample_rate: int
    voice: str
    text: Optional[str] = None


def pcm_to_wav(audio: bytes, mime_type: str, default_rate: int = 24000) -> bytes:
    """Wrap raw PCM in a WAV header.

    Streaming TTS APIs hand back headerless `audio/L16;rate=24000` chunks. The
    rate and bit depth live in the mime type, so they are read from there rather
    than assumed — a backend that switches to 16 kHz keeps working.
    """
    bits_per_sample, rate = 16, default_rate
    for param in mime_type.split(";"):
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate = int(param.split("=", 1)[1])
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass

    channels = 1
    bytes_per_sample = bits_per_sample // 8
    block_align = channels * bytes_per_sample
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(audio), b"WAVE", b"fmt ", 16, 1,
        channels, rate, rate * block_align, block_align, bits_per_sample,
        b"data", len(audio),
    )
    return header + audio


def wav_sample_rate(wav: bytes) -> Optional[int]:
    if len(wav) < 28 or wav[:4] != b"RIFF":
        return None
    return struct.unpack("<I", wav[24:28])[0]


class TTSBackend:
    """One utterance in, one clip out. Everything else is the runner's job."""

    name = "base"

    def __init__(self, config, language):
        self.config = config
        self.language = language

    async def synth(self, text: str, voice: str) -> Clip:
        raise NotImplementedError

    async def generate_text(self, prompt: str, voice: Optional[str] = None) -> str:
        raise NotImplementedError(f"{self.name} does not implement generate_text")

    async def transcribe(self, audio: bytes, sample_rate: Optional[int] = None) -> str:
        raise NotImplementedError(f"{self.name} does not implement transcribe")

    async def close(self) -> None:
        pass

    def describe(self) -> str:
        return f"{self.name} ({self.config.model})"
