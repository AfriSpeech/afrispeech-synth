"""Tests for text generation and transcription capture in afrispeech-synth."""
import pytest

from afrispeech_synth import generate_text, generate_text_async, resolve
from afrispeech_synth.config import RunConfig, TTSConfig
from afrispeech_synth.synth import Utterance, synthesise
from afrispeech_synth.tts.base import Clip, TTSBackend

# Ensure mock backend is registered
import mock_backend  # noqa: F401


def test_clip_holds_text():
    clip = Clip(audio=b"RIFF...", mime_type="audio/wav", sample_rate=24000, voice="Zephyr", text="Spoken transcript")
    assert clip.text == "Spoken transcript"


def test_generate_text_sync_mock():
    config = RunConfig()
    config.tts.backend = "mock"
    result = generate_text("Hello Ghana", config=config)
    assert "Mock text response for: Hello Ghana" in result


@pytest.mark.anyio
async def test_generate_text_async_mock():
    config = RunConfig()
    config.tts.backend = "mock"
    result = await generate_text_async("Prompt text", config=config)
    assert "Mock text response for: Prompt text" in result


def test_synth_records_spoken_transcript_in_metadata(tmp_path):
    config = TTSConfig(backend="mock", max_retries=1, rpm=0)
    utterances = [Utterance(index=0, text="Raw text", transcript="Normalised transcript")]
    res = synthesise(utterances, config, resolve("twi"), str(tmp_path))
    assert res["done"] == 1
    records = res["workspace"].records()
    assert len(records) == 1
    assert "transcript_spoken" in records[0]
    assert "Normalised transcript" in records[0]["transcript_spoken"]


def test_cli_text_command(capsys):
    from afrispeech_synth.cli import main

    ret = main(["text", "Say hi", "--backend", "mock"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Mock text response for: Say hi" in captured.out
