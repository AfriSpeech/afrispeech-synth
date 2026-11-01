"""afrispeech-synth — synthetic speech datasets for African languages.

    from afrispeech_synth import RunConfig, run

    config = RunConfig(language="twi", sources=["corpus:twi"])
    config.select.max_sentences = 500
    run(config)
"""
from .config import RunConfig, SelectConfig, TTSConfig, PackageConfig, load, from_dict
from .lang import Language, LanguageNotSupported, resolve
from .normalise import Normaliser
from .pipeline import run, stage_sources, stage_select, stage_synthesise, stage_package
from .select import Selection
from .text import generate_text, generate_text_async

__version__ = "0.1.0"

__all__ = [
    "RunConfig", "SelectConfig", "TTSConfig", "PackageConfig",
    "load", "from_dict", "run",
    "stage_sources", "stage_select", "stage_synthesise", "stage_package",
    "Language", "LanguageNotSupported", "resolve", "Normaliser", "Selection",
    "generate_text", "generate_text_async",
    "__version__",
]
