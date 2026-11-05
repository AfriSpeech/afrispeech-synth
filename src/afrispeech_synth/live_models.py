"""The Gemini Live models that generate speech, and how each one behaves.

The Live API also exposes `gemini-3.5-transcribe-live` and
`gemini-3.5-live-translate-preview`. Neither belongs here: the first is ASR
(speech in, text out) and the second translates rather than reads, which is the
exact failure this pipeline's system instruction exists to prevent. Only models
that speak the text you give them are listed.

The notes are measurements, not marketing. Each was probed on one sentence in
five Ghanaian languages (Twi, Ewe, Dagbani, Ga, Hausa), scoring the model's own
transcription of its audio against the sentence it was asked to read.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple

DEFAULT = "models/gemini-3.1-flash-live-preview"
DEFAULT_TRANSCRIBE = "models/gemini-3.5-transcribe-live"


class LiveModel(NamedTuple):
    name: str
    label: str
    probe: str
    note: str


MODELS: Dict[str, LiveModel] = {
    m.name: m for m in [
        LiveModel(
            "models/gemini-2.5-flash-native-audio-latest",
            "Gemini 2.5 Flash Native Audio",
            "5/5 clean",
            "Read every probe sentence back verbatim. Brighter than 3 Flash Live "
            "— 99% of its energy reaches ~9.7 kHz against ~5.6 kHz.",
        ),
        LiveModel(
            "models/gemini-3.1-flash-live-preview",
            "Gemini 3 Flash Live",
            "5/5 clean",
            "Read every probe sentence back verbatim, and a little faster. The "
            "default, and what the sample gallery is built with.",
        ),
        LiveModel(
            "models/gemini-3.8-live",
            "Gemini 3.8 Live",
            "2/5 clean",
            "Newest, but weakest here: returned no transcript for Twi and Ewe and "
            "a 0.8s truncated clip for Dagbani. Probe a language before trusting it.",
        ),
    ]
}


def names() -> List[str]:
    return list(MODELS)


def unknown(model: str) -> bool:
    """True for a model this build has not characterised — usable, not vetted."""
    return model not in MODELS
