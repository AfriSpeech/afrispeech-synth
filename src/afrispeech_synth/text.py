"""Text generation interface using Gemini and Gemini Live backends."""
from __future__ import annotations

import asyncio
from typing import Optional, Union

from . import tts as tts_registry
from .config import RunConfig, TTSConfig
from .lang import Language, resolve


async def generate_text_async(
    prompt: str,
    config: Optional[RunConfig] = None,
    language: Optional[Union[str, Language]] = None,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    voice: Optional[str] = None,
) -> str:
    """Generate text asynchronously using the configured backend (Gemini or Gemini Live)."""
    if config is None:
        config = RunConfig()

    tts_cfg = config.tts
    overrides = {}
    if backend is not None:
        overrides["backend"] = backend
    if model is not None:
        overrides["model"] = model
    if voice is not None:
        overrides["voice"] = voice
    if overrides:
        tts_cfg = TTSConfig(**{**vars(tts_cfg), **overrides})

    lang = resolve(language or config.language)
    backend_instance = tts_registry.get_backend(tts_cfg.backend, tts_cfg, lang)
    try:
        return await backend_instance.generate_text(prompt, voice=tts_cfg.voice)
    finally:
        await backend_instance.close()


def generate_text(
    prompt: str,
    config: Optional[RunConfig] = None,
    language: Optional[Union[str, Language]] = None,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    voice: Optional[str] = None,
) -> str:
    """Generate text synchronously using the configured backend."""
    return asyncio.run(
        generate_text_async(
            prompt,
            config=config,
            language=language,
            backend=backend,
            model=model,
            voice=voice,
        )
    )
