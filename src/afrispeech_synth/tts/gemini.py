"""Gemini TTS backend — the reference backend.

This is the path the Ghana Twi synthetic dataset was built on: streamed audio
from a Gemini TTS model, prompted with a short accent context plus the
normalised transcript.
"""
from __future__ import annotations

import os

from .base import Clip, RetryableTTSError, TTSBackend, TTSError, pcm_to_wav

RETRYABLE_MARKERS = ("429", "rate limit", "resource_exhausted", "quota",
                     "500", "502", "503", "504", "unavailable", "deadline",
                     "timeout", "internal error")


class GeminiTTS(TTSBackend):
    name = "gemini"

    def __init__(self, config, language):
        super().__init__(config, language)
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:      # pragma: no cover
            raise TTSError(
                "The Gemini backend needs the google-genai SDK:\n"
                "    pip install 'afrispeech-synth[gemini]'"
            ) from exc
        self._types = types

        from ..env import load as load_env

        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            load_env()
            api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise TTSError(
                f"No API key found. Either:\n"
                f"    export {config.api_key_env}=...\n"
                f"or put `{config.api_key_env}=...` in a .env file in this directory.\n"
                f"Keys must never go in a config file — those get committed."
            )
        self._client = genai.Client(api_key=api_key)

    def _request_config(self, voice: str):
        types = self._types
        return types.GenerateContentConfig(
            temperature=self.config.temperature,
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        )

    async def synth(self, text: str, voice: str) -> Clip:
        types = self._types
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=text)])]

        audio = bytearray()
        mime_type = ""
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self.config.model,
                contents=contents,
                config=self._request_config(voice),
            )
            async for chunk in stream:
                if chunk.parts and chunk.parts[0].inline_data and chunk.parts[0].inline_data.data:
                    inline = chunk.parts[0].inline_data
                    audio.extend(inline.data)
                    mime_type = inline.mime_type or mime_type
        except Exception as exc:
            message = str(exc).lower()
            if any(marker in message for marker in RETRYABLE_MARKERS):
                raise RetryableTTSError(str(exc)) from exc
            raise TTSError(str(exc)) from exc

        if not audio:
            # An empty stream is usually a safety block or a silently dropped
            # request; retrying it often succeeds, so treat it as transient.
            raise RetryableTTSError("model returned no audio")

        data = bytes(audio)
        if not data.startswith(b"RIFF"):
            data = pcm_to_wav(data, mime_type, default_rate=self.config.sample_rate)
        return Clip(audio=data, mime_type="audio/wav", voice=voice,
                    sample_rate=self.config.sample_rate)

    async def generate_text(self, prompt: str, voice: Optional[str] = None) -> str:
        types = self._types
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        try:
            response = await self._client.aio.models.generate_content(
                model=self.config.model,
                contents=contents,
            )
            text = response.text or ""
            if not text and response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    text = "".join(p.text for p in candidate.content.parts if p.text)
            return text.strip()
        except Exception as exc:
            message = str(exc).lower()
            if any(marker in message for marker in RETRYABLE_MARKERS):
                raise RetryableTTSError(str(exc)) from exc
            raise TTSError(str(exc)) from exc
