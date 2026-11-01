"""A TTS backend that generates silence, so the pipeline can be tested offline."""
import math
import struct

from afrispeech_synth import tts as tts_registry
from afrispeech_synth.tts.base import Clip, TTSBackend, pcm_to_wav


class MockTTS(TTSBackend):
    name = "mock"

    async def synth(self, text: str, voice: str) -> Clip:
        rate = self.config.sample_rate
        samples = int(rate * 0.2)
        pcm = b"".join(
            struct.pack("<h", int(3000 * math.sin(2 * math.pi * 220 * i / rate)))
            for i in range(samples)
        )
        return Clip(audio=pcm_to_wav(pcm, f"audio/L16;rate={rate}", rate),
                    mime_type="audio/wav", sample_rate=rate, voice=voice, text=text)

    async def generate_text(self, prompt: str, voice: str = "Zephyr") -> str:
        return f"Mock text response for: {prompt}"


tts_registry.register("mock", lambda: MockTTS)
