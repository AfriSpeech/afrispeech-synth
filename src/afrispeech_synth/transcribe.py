"""Audio transcription interface using Gemini Live models.

This module provides real-time speech-to-text powered strictly by Gemini Live
WebSocket models, primarily `models/gemini-3.5-transcribe-live`.
"""
from __future__ import annotations

import asyncio
import io
import os
import struct
import wave
from typing import Callable, List, Optional, Sequence, Tuple, Union

from .config import RunConfig, TTSConfig
from .lang import Language, resolve, resolve_transcribe_language_codes

DEFAULT_TRANSCRIBE_MODEL = "models/gemini-3.5-transcribe-live"
DEFAULT_SAMPLE_RATE = 24000


def is_live_model(model: str) -> bool:
    """Return True if the model name indicates a Gemini Live model."""
    m = (model or "").lower()
    return "live" in m or "native-audio" in m


def validate_live_model(model: str) -> str:
    """Ensure strictly that only Live models are used for transcription."""
    if not is_live_model(model):
        raise ValueError(
            f"Model {model!r} is not a Gemini Live model. Only Gemini Live models "
            f"(such as 'models/gemini-3.5-transcribe-live') are supported for real-time transcription."
        )
    if not model.startswith("models/") and "/" not in model:
        return f"models/{model}"
    return model


def _to_mono_16bit(pcm: bytes, nchannels: int, sampwidth: int) -> bytes:
    """Convert multi-channel or non-16bit PCM to single-channel 16-bit PCM."""
    if nchannels == 1 and sampwidth == 2:
        return pcm

    if sampwidth == 2 and nchannels > 1:
        out = bytearray()
        fmt = f"<{nchannels}h"
        step = 2 * nchannels
        for i in range(0, len(pcm) - step + 1, step):
            samples = struct.unpack_from(fmt, pcm, i)
            avg = sum(samples) // nchannels
            out.extend(struct.pack("<h", avg))
        return bytes(out)

    return pcm


def read_audio_pcm(
    audio: Union[str, os.PathLike, bytes, bytearray, io.IOBase],
    default_rate: int = DEFAULT_SAMPLE_RATE,
) -> Tuple[bytes, int]:
    """Extract raw 16-bit mono PCM bytes and sample rate from a file path or bytes buffer."""
    import shutil
    import subprocess

    if isinstance(audio, (str, os.PathLike)):
        path = str(audio)
        if path.lower().endswith(".wav"):
            try:
                with wave.open(path, "rb") as wf:
                    rate = wf.getframerate()
                    nchannels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    frames = wf.readframes(wf.getnframes())
                    return _to_mono_16bit(frames, nchannels, sampwidth), rate
            except Exception:
                pass

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            proc = subprocess.run(
                [ffmpeg_bin, "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(default_rate), "pipe:1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return proc.stdout, default_rate

        with wave.open(path, "rb") as wf:
            rate = wf.getframerate()
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
            return _to_mono_16bit(frames, nchannels, sampwidth), rate

    if isinstance(audio, (bytes, bytearray)):
        if audio.startswith(b"RIFF"):
            try:
                with wave.open(io.BytesIO(audio), "rb") as wf:
                    rate = wf.getframerate()
                    nchannels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    frames = wf.readframes(wf.getnframes())
                    return _to_mono_16bit(frames, nchannels, sampwidth), rate
            except Exception:
                pass

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin and (audio.startswith(b"ID3") or audio.startswith(b"\xff\xfb") or audio.startswith(b"\xff\xf3") or audio.startswith(b"OggS")):
            proc = subprocess.run(
                [ffmpeg_bin, "-v", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", str(default_rate), "pipe:1"],
                input=bytes(audio),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return proc.stdout, default_rate

        # Raw PCM fallback
        return bytes(audio), default_rate

    if hasattr(audio, "read"):
        data = audio.read()
        return read_audio_pcm(data, default_rate=default_rate)

    raise TypeError(f"Unsupported audio input type: {type(audio)}")


async def transcribe_audio_async(
    audio: Union[str, os.PathLike, bytes, bytearray, io.IOBase],
    language: Optional[Union[str, Language]] = None,
    language_codes: Optional[Sequence[str]] = None,
    model: str = DEFAULT_TRANSCRIBE_MODEL,
    custom_vocabulary: Optional[Sequence[str]] = None,
    mode: Optional[str] = None,
    config: Optional[RunConfig] = None,
    backend: Optional[str] = None,
    api_key_env: str = "GEMINI_API_KEY",
    timeout: float = 30.0,
    on_interim: Optional[Callable[[str], None]] = None,
) -> str:
    """Transcribe audio asynchronously using the Gemini Live API.

    Parameters:
        audio: Audio file path, WAV bytes, or raw PCM bytes.
        language: Language name or code to resolve into BCP-47 hints (e.g. "twi", "yoruba").
        language_codes: Explicit BCP-47 language codes override (e.g. ["ak-GH", "ak"]).
        model: Live model to use (must be a live model, default: 'models/gemini-3.5-transcribe-live').
        custom_vocabulary: Optional list of phrases or words to bias the ASR engine.
        mode: Optional transcription mode ('VERBATIM' or 'SMART').
        config: Optional RunConfig (if backend overrides are used, e.g. for testing).
        backend: Optional backend override (e.g. 'mock' in tests).
        api_key_env: Environment variable holding the Gemini API key.
        timeout: Maximum seconds to wait for transcription.
        on_interim: Optional callback invoked with low-latency interim transcription updates.
    """
    # Check for mock or custom test backends
    if backend and backend != "gemini-live":
        from . import tts as tts_registry
        tts_cfg = (config.tts if config else TTSConfig())
        if isinstance(language, Language):
            lang_obj = language
        elif language:
            lang_obj = resolve(language)
        else:
            lang_obj = resolve(tts_cfg.voice or "english")
        backend_instance = tts_registry.get_backend(backend, tts_cfg, lang_obj)
        raw_pcm, rate = read_audio_pcm(audio)
        return await backend_instance.transcribe(raw_pcm, sample_rate=rate)

    # Strictly enforce Live models
    resolved_model = validate_live_model(model)

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The Gemini Live transcription backend needs the google-genai SDK:\n"
            "    pip install 'afrispeech-synth[gemini]'"
        ) from exc

    from .env import load as load_env

    api_key = os.environ.get(api_key_env)
    if not api_key:
        load_env()
        api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(
            f"No API key found in environment variable {api_key_env}.\n"
            f"Set it with: export {api_key_env}=... or place it in a .env file."
        )

    # Determine BCP-47 language codes
    hints: List[str] = []
    if language_codes:
        hints = list(language_codes)
    elif language:
        hints = resolve_transcribe_language_codes(language)

    # Configure Live AudioTranscriptionConfig
    asr_kwargs = {}
    if hints:
        asr_kwargs["language_codes"] = hints
    if custom_vocabulary:
        asr_kwargs["custom_vocabulary"] = list(custom_vocabulary)
    if mode:
        mode_val = mode.upper()
        if hasattr(types.AudioTranscriptionConfigMode, mode_val):
            asr_kwargs["mode"] = getattr(types.AudioTranscriptionConfigMode, mode_val)

    # For transcribe-live model, text modality is used; for conversational live models, audio is required.
    response_modalities = (
        [types.Modality.TEXT]
        if "transcribe" in resolved_model
        else [types.Modality.AUDIO]
    )

    connect_config = types.LiveConnectConfig(
        response_modalities=response_modalities,
        input_audio_transcription=types.AudioTranscriptionConfig(**asr_kwargs),
    )

    raw_pcm, rate = read_audio_pcm(audio)
    duration = len(raw_pcm) / (rate * 2) if rate else 0
    effective_timeout = max(timeout, duration + 30.0)
    client = genai.Client(api_key=api_key)

    async with client.aio.live.connect(model=resolved_model, config=connect_config) as session:
        chunk_size = int(rate * 2 * 0.1)
        audio_done = asyncio.Event()

        async def _send_audio():
            for i in range(0, len(raw_pcm), chunk_size):
                chunk = raw_pcm[i : i + chunk_size]
                await session.send_realtime_input(
                    audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={rate}")
                )
                await asyncio.sleep(0.01)
            await session.send_realtime_input(audio_stream_end=True)
            audio_done.set()

        transcript_parts: List[str] = []

        async def _listen():
            async for msg in session.receive():
                sc = msg.server_content
                if not sc:
                    continue
                if (
                    on_interim
                    and sc.interim_input_transcription
                    and sc.interim_input_transcription.text
                ):
                    on_interim(sc.interim_input_transcription.text)
                if sc.input_transcription and sc.input_transcription.text:
                    transcript_parts.append(sc.input_transcription.text)
                if audio_done.is_set() and (sc.generation_complete or sc.turn_complete):
                    break

        sender_task = asyncio.create_task(_send_audio())
        listener_task = asyncio.create_task(_listen())
        try:
            await asyncio.wait_for(asyncio.gather(sender_task, listener_task), timeout=effective_timeout)
        except asyncio.TimeoutError:
            sender_task.cancel()
            listener_task.cancel()

        return " ".join(transcript_parts).strip()


def transcribe_audio(
    audio: Union[str, os.PathLike, bytes, bytearray, io.IOBase],
    language: Optional[Union[str, Language]] = None,
    language_codes: Optional[Sequence[str]] = None,
    model: str = DEFAULT_TRANSCRIBE_MODEL,
    custom_vocabulary: Optional[Sequence[str]] = None,
    mode: Optional[str] = None,
    config: Optional[RunConfig] = None,
    backend: Optional[str] = None,
    api_key_env: str = "GEMINI_API_KEY",
    timeout: float = 30.0,
    on_interim: Optional[Callable[[str], None]] = None,
) -> str:
    """Synchronously transcribe audio using the Gemini Live API."""
    return asyncio.run(
        transcribe_audio_async(
            audio=audio,
            language=language,
            language_codes=language_codes,
            model=model,
            custom_vocabulary=custom_vocabulary,
            mode=mode,
            config=config,
            backend=backend,
            api_key_env=api_key_env,
            timeout=timeout,
            on_interim=on_interim,
        )
    )
