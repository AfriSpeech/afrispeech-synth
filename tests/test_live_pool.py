"""The Live backend must keep open websockets within `tts.concurrency`.

Projects have a concurrent-session quota on the Live API. A per-voice pool that
never closes idle sessions grows with every voice in a rotation and then fails
every connect with "1011 ... exceeded your current quota". This drives the real
backend through the real runner with a fake Live client that counts open sockets.
"""
import asyncio

import pytest

from afrispeech_synth import resolve
from afrispeech_synth import tts as tts_registry
from afrispeech_synth.config import TTSConfig
from afrispeech_synth.synth import Utterance, synthesise

gemini_live = pytest.importorskip("afrispeech_synth.tts.gemini_live")


class _Counter:
    def __init__(self):
        self.open = 0
        self.peak = 0
        self.connects = 0


class _FakeConnect:
    """Stands in for `client.aio.live.connect(...)`: an async context manager."""

    def __init__(self, counter):
        self.counter = counter

    async def __aenter__(self):
        self.counter.open += 1
        self.counter.connects += 1
        self.counter.peak = max(self.counter.peak, self.counter.open)
        return object()

    async def __aexit__(self, *exc):
        self.counter.open -= 1


class FakeLiveTTS(gemini_live.GeminiLiveTTS):
    counter = None

    def __init__(self, config, language):          # skip the SDK and API key
        self.config, self.language = config, language
        self._idle, self._open, self._lock = {}, [], asyncio.Lock()
        self._max_open = max(1, int(config.concurrency or 1))
        self._model_label = "fake"
        counter = FakeLiveTTS.counter

        class _Live:
            @staticmethod
            def connect(model, config):
                return _FakeConnect(counter)

        class _Aio:
            live = _Live()

        class _Client:
            aio = _Aio()

        self._client = _Client()

    def _connect_config(self, voice):
        return None

    async def _turn(self, session, text):
        await asyncio.sleep(0.001)
        return b"\x00\x00" * 2400, text, "audio/pcm;rate=24000"


@pytest.fixture
def fake_live():
    FakeLiveTTS.counter = _Counter()
    tts_registry.register("fake-live", lambda: FakeLiveTTS)
    return FakeLiveTTS.counter


VOICES = ["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
          "Callirrhoe", "Iapetus"]


def _run(tmp_path, utterances, concurrency):
    config = TTSConfig(backend="fake-live", voices=VOICES, concurrency=concurrency,
                       rpm=0, max_retries=1)
    return synthesise(utterances, config, resolve("twi"), str(tmp_path))


def test_open_sessions_never_exceed_concurrency(tmp_path, fake_live):
    utterances = [Utterance(index=i, text=f"s{i}", transcript=f"s{i}",
                            voice=VOICES[i % len(VOICES)]) for i in range(60)]
    result = _run(tmp_path, utterances, concurrency=3)
    assert result["done"] == 60 and result["failed"] == 0
    assert fake_live.peak <= 3, f"{fake_live.peak} sessions open at once"
    assert fake_live.open == 0, "close() must release every session"


def test_voices_are_grouped_so_sessions_are_reused(tmp_path, fake_live):
    # 60 clips over 10 voices: grouped by voice, each voice needs only a few
    # connects instead of roughly one per clip.
    utterances = [Utterance(index=i, text=f"s{i}", transcript=f"s{i}",
                            voice=VOICES[i % len(VOICES)]) for i in range(60)]
    _run(tmp_path, utterances, concurrency=3)
    assert fake_live.connects <= 3 * len(VOICES), fake_live.connects
