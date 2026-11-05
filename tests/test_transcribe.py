"""Tests for real-time audio transcription with Gemini Live API."""
import io
import math
import struct
import wave
import pytest

from afrispeech_synth import (
    resolve,
    resolve_transcribe_language_codes,
    transcribe_audio,
    transcribe_audio_async,
)
from afrispeech_synth.config import RunConfig, TTSConfig
from afrispeech_synth.synth import Utterance, synthesise
from afrispeech_synth.transcribe import (
    DEFAULT_TRANSCRIBE_MODEL,
    is_live_model,
    read_audio_pcm,
    validate_live_model,
)

# Ensure mock backend is registered
import mock_backend  # noqa: F401


def _generate_test_wav(duration: float = 0.2, rate: int = 16000, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        total_samples = int(duration * rate)
        pcm = bytearray()
        for i in range(total_samples):
            val = int(3000 * math.sin(2 * math.pi * 440 * i / rate))
            pcm.extend(struct.pack("<h", val) * channels)
        wf.writeframes(bytes(pcm))
    return buf.getvalue()


def test_language_code_resolution():
    twi_codes = resolve_transcribe_language_codes("twi")
    assert "ak-GH" in twi_codes
    assert "ak" in twi_codes

    yoruba_codes = resolve_transcribe_language_codes("yoruba")
    assert "yo-NG" in yoruba_codes
    assert "yo" in yoruba_codes

    swahili_codes = resolve_transcribe_language_codes("swahili")
    assert "sw-KE" in swahili_codes or "sw" in swahili_codes

    hausa_codes = resolve_transcribe_language_codes("hausa")
    assert "ha-NG" in hausa_codes or "ha" in hausa_codes

    # Language object input
    lang_obj = resolve("ewe")
    ewe_codes = resolve_transcribe_language_codes(lang_obj)
    assert "ee-GH" in ewe_codes or "ee" in ewe_codes

    # Unknown language code fallback
    unknown_codes = resolve_transcribe_language_codes("xyz")
    assert "xyz" in unknown_codes


def test_validate_live_model_strictly_enforces_live():
    # Valid live models
    assert validate_live_model("gemini-3.5-transcribe-live") == "models/gemini-3.5-transcribe-live"
    assert validate_live_model("models/gemini-3.5-transcribe-live") == "models/gemini-3.5-transcribe-live"
    assert validate_live_model("models/gemini-3.1-flash-live-preview") == "models/gemini-3.1-flash-live-preview"
    assert validate_live_model("gemini-2.5-flash-native-audio-latest") == "models/gemini-2.5-flash-native-audio-latest"

    # Non-live models must be rejected
    for non_live in [
        "gemini-2.5-flash",
        "gemini-3.5-transcribe",  # REST / non-live
        "gemini-3.1-flash-tts-preview",
        "gpt-4o",
    ]:
        assert not is_live_model(non_live)
        with pytest.raises(ValueError, match="not a Gemini Live model"):
            validate_live_model(non_live)


def test_read_audio_pcm_wav_mono():
    wav_bytes = _generate_test_wav(duration=0.1, rate=16000, channels=1)
    pcm, rate = read_audio_pcm(wav_bytes)
    assert rate == 16000
    assert len(pcm) == int(0.1 * 16000 * 2)


def test_read_audio_pcm_wav_stereo():
    wav_bytes = _generate_test_wav(duration=0.1, rate=24000, channels=2)
    pcm, rate = read_audio_pcm(wav_bytes)
    assert rate == 24000
    # Downmixed mono should be half the stereo payload
    assert len(pcm) == int(0.1 * 24000 * 2)


def test_read_audio_pcm_raw_pcm():
    raw = b"\x00\x00" * 100
    pcm, rate = read_audio_pcm(raw, default_rate=24000)
    assert pcm == raw
    assert rate == 24000


def test_transcribe_audio_sync_mock(tmp_path):
    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(_generate_test_wav())

    result = transcribe_audio(str(audio_path), backend="mock")
    assert "Mock transcript for audio" in result


@pytest.mark.anyio
async def test_transcribe_audio_async_mock(tmp_path):
    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(_generate_test_wav())

    result = await transcribe_audio_async(str(audio_path), backend="mock")
    assert "Mock transcript for audio" in result


def test_cli_transcribe_command(capsys, tmp_path):
    from afrispeech_synth.cli import main

    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(_generate_test_wav())

    ret = main(["transcribe", str(audio_path), "--backend", "mock", "--lang", "twi"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Mock transcript for audio" in captured.out


def test_synth_with_transcribe_flag_records_transcript(tmp_path):
    config = TTSConfig(backend="mock", max_retries=1, rpm=0, transcribe=True)
    utterances = [Utterance(index=0, text="Raw text", transcript="Normalised transcript")]
    res = synthesise(utterances, config, resolve("twi"), str(tmp_path))
    assert res["done"] == 1
    records = res["workspace"].records()
    assert len(records) == 1
    assert "transcript_transcribed" in records[0]
    assert "Mock transcript for audio" in records[0]["transcript_transcribed"]
