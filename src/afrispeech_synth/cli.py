"""Command line interface.

    afrispeech-synth run config.yaml            # the whole pipeline
    afrispeech-synth run config.yaml --dry-run  # select sentences only
    afrispeech-synth select --lang twi --source corpus:twi --max-sentences 2000
    afrispeech-synth synth  config.yaml         # resume synthesis only
    afrispeech-synth package config.yaml        # rebuild parquet from the work dir
    afrispeech-synth push    config.yaml --repo AfriSpeech/twi-synthetic-speech
    afrispeech-synth langs --search yor
"""
from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from . import card as card_module
from . import config as config_module
from . import package as package_module
from . import pipeline
from . import publish as publish_module
from . import tts as tts_registry
from .lang import resolve
from .normalise import Normaliser


def _config_from_args(args) -> config_module.RunConfig:
    """A config file, CLI flags, or both — flags always win."""
    if getattr(args, "config", None):
        config = config_module.load(args.config)
    else:
        config = config_module.RunConfig()

    overrides = {}
    for flag, target in [
        ("lang", "language"), ("out", "out"), ("work", "work"),
        ("normalise", "normalise"),
        ("cover", "select.cover"), ("min_freq", "select.min_freq"),
        ("max_sentences", "select.max_sentences"),
        ("min_chars", "select.min_chars"), ("max_chars", "select.max_chars"),
        ("backend", "tts.backend"), ("model", "tts.model"),
        ("concurrency", "tts.concurrency"), ("rpm", "tts.rpm"),
        ("repo", "package.push_to"),
    ]:
        value = getattr(args, flag, None)
        if value is not None:
            overrides[target] = value
    if getattr(args, "voice", None):
        overrides["tts.voice"] = args.voice.strip()
    if getattr(args, "voices", None):
        overrides["tts.voices"] = [v.strip() for v in args.voices.split(",") if v.strip()]
    if getattr(args, "source", None):
        config.sources = list(args.source)
    if getattr(args, "format", None):
        overrides["package.formats"] = [f.strip() for f in args.format.split(",") if f.strip()]

    config_module.apply_overrides(config, overrides)
    if not config.sources and getattr(args, "_needs_sources", True):
        config.sources = [f"corpus:{resolve(config.language).code}"]
    return config


def cmd_run(args) -> int:
    config = _config_from_args(args)
    pipeline.run(config, resume=not args.no_resume, dry_run=args.dry_run)
    return 0


def cmd_select(args) -> int:
    config = _config_from_args(args)
    language = resolve(config.language)
    normaliser = Normaliser(language, config.normalise)
    sentences = pipeline.stage_sources(config, language)
    selection = pipeline.stage_select(config, language, sentences, normaliser)
    pipeline._save_sentences(config, selection.sentences, selection)
    out = os.path.join(config.work_dir, pipeline.SENTENCES_FILE)
    print(f"\n{len(selection.sentences)} sentences -> {out}")
    return 0


def cmd_synth(args) -> int:
    config = _config_from_args(args)
    language = resolve(config.language)
    sentences = pipeline.load_sentences(config)
    if sentences is None:
        print(f"No {pipeline.SENTENCES_FILE} in {config.work_dir}. Run `select` first, "
              f"or use `run` to do everything.", file=sys.stderr)
        return 1
    normaliser = Normaliser(language, config.normalise)
    pipeline.stage_synthesise(config, language, sentences, normaliser,
                              resume=not args.no_resume)
    return 0


def cmd_package(args) -> int:
    config = _config_from_args(args)
    language = resolve(config.language)
    pipeline.stage_package(config, language)
    return 0


def cmd_push(args) -> int:
    config = _config_from_args(args)
    repo = args.repo or config.package.push_to
    if not repo:
        print("No target repo. Pass --repo org/name or set package.push_to.", file=sys.stderr)
        return 1
    publish_module.push(config.out, repo, private=config.package.private)
    return 0


def cmd_card(args) -> int:
    config = _config_from_args(args)
    language = resolve(config.language)
    clips = len(package_module.Workspace(config.work_dir).records()) if \
        os.path.exists(config.work_dir) else 0
    print(card_module.render(config, language, clips))
    return 0


def cmd_text(args) -> int:
    from .text import generate_text
    prompt = getattr(args, "prompt", None)
    if not prompt:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        else:
            print("error: a prompt argument or piped input is required", file=sys.stderr)
            return 1

    config = _config_from_args(args) if getattr(args, "config", None) else None
    result = generate_text(
        prompt=prompt,
        config=config,
        language=getattr(args, "lang", None),
        backend=getattr(args, "backend", None),
        model=getattr(args, "model", None),
        voice=getattr(args, "voice", None),
    )
    print(result)
    return 0


STATUS_HELP = {
    "ready": "text + G2P available — just name the language",
    "bring text": "G2P available; supply text with a file: or hf: source",
    "no g2p": "text available; run with --normalise none",
}


def cmd_langs(args) -> int:
    from . import coverage

    catalogue = coverage.load()
    entries = catalogue.search(args.search) if args.search else catalogue.sorted()
    if args.ready:
        entries = [e for e in entries if e.ready]

    for entry in entries:
        print(f"{entry.code:<6} {entry.name[:34]:<35} {entry.status:<11} {entry.family}")

    counts = catalogue.counts()
    print(f"\n{len(entries)} shown.", file=sys.stderr)
    print(f"  {counts['ready']:>4} ready       {STATUS_HELP['ready']}", file=sys.stderr)
    print(f"  {counts['g2p'] - counts['ready']:>4} bring text  {STATUS_HELP['bring text']}",
          file=sys.stderr)
    print(f"  {counts['text'] - counts['ready']:>4} no g2p      {STATUS_HELP['no g2p']}",
          file=sys.stderr)
    if not counts["text"]:
        print("\n  (africa-corpus-builder not found, so no language shows as ready — "
              "see the README to install it.)", file=sys.stderr)
    return 0


def cmd_samples(args) -> int:
    from . import samples as samples_module

    config = _config_from_args(args)
    codes = [c.strip() for c in args.langs.split(",") if c.strip()] if args.langs else None
    built = samples_module.build(config, args.dir, codes=codes, limit=args.limit,
                                 resume=not args.no_resume,
                                 all_voices=args.all_voices,
                                 distinct=args.distinct,
                                 compress=args.compress)
    print(f"\n{len(built)} samples in {args.dir}. Next: afrispeech-synth space "
          f"--dir {args.dir} --repo org/name")
    return 0


def cmd_space(args) -> int:
    from . import samples as samples_module
    from . import space as space_module

    args.title = args.title or space_module.DEFAULT_TITLE

    built = samples_module.load_manifest(args.dir)
    if not built:
        print(f"No samples.json in {args.dir}. Run `samples` first.", file=sys.stderr)
        return 1
    # The dataset goes first: the page needs its URLs baked in before it is built.
    audio_base = None
    if args.audio_repo:
        from . import gallery_dataset
        gallery_dataset.build(args.dir, args.audio_repo,
                              gallery_dataset.DEFAULT_TITLE)
        audio_base = gallery_dataset.push(args.dir, args.audio_repo,
                                          private=args.private)
    space_module.build(built, args.dir, title=args.title, audio_base=audio_base)
    if args.repo:
        space_module.push(args.dir, args.repo, private=args.private,
                          audio_elsewhere=bool(audio_base))
    else:
        print(f"\nOpen {os.path.join(args.dir, 'index.html')} to preview, then re-run "
              f"with --repo org/name to publish.")
    return 0


def cmd_models(args) -> int:
    from .live_models import DEFAULT, MODELS

    print("Gemini Live models that generate speech — set one as `tts.model` "
          "with `tts.backend: gemini-live`.\n")
    print(f"  {'model':46s} {'probe':10s} {'name'}")
    for model in MODELS.values():
        mark = " *" if model.name == DEFAULT else "  "
        print(f"{mark}{model.name:46s} {model.probe:10s} {model.label}")
        print(f"   {'':46s} {'':10s} {model.note}")
    print("\n  * default. Probe = one sentence in each of Twi, Ewe, Dagbani, Ga and "
          "Hausa,\n    scored against what the model said it spoke.", file=sys.stderr)
    print("  The TTS backend (`tts.backend: gemini`) is separate: "
          "gemini-3.1-flash-tts-preview.", file=sys.stderr)
    return 0


def cmd_voices(args) -> int:
    from .voices import GEMINI_VOICES

    print("Gemini voices — set one as `tts.voice` for a run. Hear them first "
          "in the samples gallery, then pick:\n")
    for name, character in GEMINI_VOICES.items():
        print(f"  {name:<16} {character}")
    print(f"\n{len(GEMINI_VOICES)} voices. Example:  --voice Sulafat", file=sys.stderr)
    return 0


def cmd_init(args) -> int:
    language = resolve(args.lang)
    config = config_module.RunConfig(
        language=language.code,
        sources=[f"corpus:{language.code}"],
        out=args.out or f"out/{language.code}",
    )
    config.select.max_sentences = 2000
    config.tts.context = f"speak in {language.name} accent"
    path = args.config or f"{language.code}.yaml"
    config_module.dump(config, path)
    print(f"Wrote {path}. Edit it, then: afrispeech-synth run {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="afrispeech-synth",
        description="Generate synthetic speech datasets for African languages.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub, with_config=True):
        if with_config:
            sub.add_argument("config", nargs="?", help="YAML run config")
        sub.add_argument("--lang", help="Language name or code (e.g. Twi, twi, yor)")
        sub.add_argument("--source", action="append",
                         help="Text source URI; repeatable (corpus:twi, hf:org/ds#col, file:x.txt)")
        sub.add_argument("--out", help="Output directory")
        sub.add_argument("--work", help="Work directory (default: <out>/work)")
        sub.add_argument("--normalise", choices=["grapheme", "universal", "ipa", "none"])
        sub.add_argument("--cover", choices=["phoneme", "word", "none"])
        sub.add_argument("--min-freq", type=int)
        sub.add_argument("--max-sentences", type=int)
        sub.add_argument("--min-chars", type=int)
        sub.add_argument("--max-chars", type=int)
        sub.add_argument("--backend", choices=tts_registry.available())
        sub.add_argument("--model")
        sub.add_argument("--voice", help="Voice for this dataset (default Zephyr)")
        sub.add_argument("--concurrency", type=int)
        sub.add_argument("--rpm", type=int)
        sub.add_argument("--format", help="Comma-separated: parquet,ljspeech")
        sub.add_argument("--repo", help="HuggingFace dataset repo to push to")
        return sub

    run_parser = add_common(subparsers.add_parser("run", help="Run the whole pipeline"))
    run_parser.add_argument("--dry-run", action="store_true",
                            help="Select sentences and stop, without calling the TTS API")
    run_parser.add_argument("--no-resume", action="store_true",
                            help="Ignore existing work and start over")
    run_parser.set_defaults(func=cmd_run)

    add_common(subparsers.add_parser("select", help="Source + select sentences only")
               ).set_defaults(func=cmd_select)

    synth_parser = add_common(subparsers.add_parser("synth", help="Synthesise selected sentences"))
    synth_parser.add_argument("--no-resume", action="store_true")
    synth_parser.set_defaults(func=cmd_synth)

    add_common(subparsers.add_parser("package", help="Build parquet + manifest + card")
               ).set_defaults(func=cmd_package)
    add_common(subparsers.add_parser("push", help="Upload a packaged dataset to the Hub")
               ).set_defaults(func=cmd_push)
    add_common(subparsers.add_parser("card", help="Print the dataset card")
               ).set_defaults(func=cmd_card)

    text_parser = subparsers.add_parser("text", help="Generate text using Gemini or Gemini Live")
    text_parser.add_argument("prompt", nargs="?", help="Prompt for text generation (or pipe via stdin)")
    text_parser.add_argument("--config", help="Optional YAML config")
    text_parser.add_argument("--lang", default="twi", help="Language name or code (default: twi)")
    text_parser.add_argument("--backend", choices=tts_registry.available(), help="Backend: gemini or gemini-live")
    text_parser.add_argument("--model", help="Model name (e.g. gemini-3.8-live, gemini-3.1-flash-live-preview)")
    text_parser.add_argument("--voice", help="Voice for Live session (default: Zephyr)")
    text_parser.set_defaults(func=cmd_text)

    init_parser = subparsers.add_parser("init", help="Write a starter config for a language")
    init_parser.add_argument("lang", help="Language name or code")
    init_parser.add_argument("--config", help="Config path to write")
    init_parser.add_argument("--out", help="Output directory to record in the config")
    init_parser.set_defaults(func=cmd_init)

    langs_parser = subparsers.add_parser(
        "langs", help="List languages and what each one needs to run")
    langs_parser.add_argument("--search", help="Filter by code or name")
    langs_parser.add_argument("--ready", action="store_true",
                              help="Only languages that run with no text of your own")
    langs_parser.set_defaults(func=cmd_langs)

    voices_parser = subparsers.add_parser("voices", help="List the TTS voices you can choose")
    voices_parser.set_defaults(func=cmd_voices)

    models_parser = subparsers.add_parser(
        "models", help="List the Gemini Live models you can synthesise with")
    models_parser.set_defaults(func=cmd_models)

    samples_parser = add_common(
        subparsers.add_parser("samples", help="Generate one sample clip per language"))
    samples_parser.add_argument("--dir", default="space",
                                help="Where samples and the gallery are built (default: space/)")
    samples_parser.add_argument("--langs", help="Comma-separated codes (default: every ready language)")
    samples_parser.add_argument("--limit", type=int, help="Stop after this many languages")
    samples_parser.add_argument("--no-resume", action="store_true")
    samples_parser.add_argument("--all-voices", action="store_true",
                                help="Every voice for every language, not one each")
    samples_parser.add_argument("--distinct", action="store_true",
                                help="With --all-voices, give each voice its own "
                                     "sentence instead of repeating one. Makes a small "
                                     "dataset rather than a voice comparison.")
    samples_parser.add_argument("--compress", action="store_true",
                                help="Re-encode clips to 64k mono MP3. Only worth it when "
                                     "the audio ships inside the Space; with --audio-repo "
                                     "the dataset should keep the original WAV.")
    samples_parser.add_argument("--voices",
                                help="Comma-separated voice pool to spread across "
                                     "languages (default: all 30)")
    samples_parser.set_defaults(func=cmd_samples)

    space_parser = subparsers.add_parser(
        "space", help="Build the samples gallery page, and optionally push it as a HF Space")
    space_parser.add_argument("--dir", default="space", help="Directory holding samples.json")
    space_parser.add_argument("--repo", help="HuggingFace Space to push to, e.g. AfriSpeech/samples")
    space_parser.add_argument("--title", default=None,
                              help="Page title (default names it as synthetic speech)")
    space_parser.add_argument("--audio-repo",
                              help="Dataset repo to host the clips in, e.g. "
                                   "AfriSpeech/synthetic-voice-samples-africa. The Space "
                                   "then streams from it instead of carrying hundreds of "
                                   "MB itself.")
    space_parser.add_argument("--private", action="store_true")
    space_parser.set_defaults(func=cmd_space)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
