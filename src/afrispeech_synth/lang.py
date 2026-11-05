"""Language resolution.

Three libraries in this stack name languages differently: a user types "Twi",
africa-g2p wants a code in its 400-language registry, africa-corpus-builder
wants a code in its 693-language registry. afriso maps 2,264 African language
names, alternative names and ISO 639-1/2/3 codes onto one ISO 639-3 code, so it
is the single place that handles the naming mess.

afriso is optional. Without it a token is passed straight through to the other
libraries, which still works whenever the user already typed a valid code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

try:  # pragma: no cover - exercised by whichever env the user has
    import afriso as _afriso
except ImportError:  # pragma: no cover
    _afriso = None


_NAME_INDEX = None


class LanguageNotSupported(Exception):
    pass


@dataclass
class Language:
    """One language, resolved once and reused by every stage of a run."""

    token: str                      # whatever the user typed
    code: str                       # ISO 639-3 where afriso could resolve it
    name: str                       # display name, for the dataset card
    g2p_code: Optional[str] = None  # code africa-g2p accepts, or None
    family: Optional[str] = None
    countries: tuple = field(default_factory=tuple)

    @property
    def has_g2p(self) -> bool:
        return self.g2p_code is not None


_G2P_REGISTRY = None


def _g2p_registry() -> dict:
    """Every language africa-g2p can convert, with metadata where it exists.

    `registry()` is the metadata table and `available_languages()` is what can
    actually be converted; tables have been added faster than metadata rows, so
    the two differ by hundreds. Reading only the registry meant a language with
    a rule table but no description of itself was reported as unsupported.
    """
    global _G2P_REGISTRY
    if _G2P_REGISTRY is None:
        from africa_g2p import available_languages, registry
        meta = registry()
        _G2P_REGISTRY = {code: (meta.get(code) or {"iso639_3": code})
                         for code in available_languages()}
        # Keep metadata-only rows too: they may carry aliases worth resolving.
        for code, entry in meta.items():
            _G2P_REGISTRY.setdefault(code, entry)
    return _G2P_REGISTRY


def _g2p_name_index() -> dict:
    """name / alt-name -> code, from africa-g2p's own registry.

    africa-g2p's registry was itself enriched from afriso, so it carries names,
    alternative names and families for its 400 languages. Indexing it means
    `--lang Yoruba` works even when afriso is not installed; afriso still adds
    the other ~1,860 languages, which matters for the corpus side.
    """
    global _NAME_INDEX
    if _NAME_INDEX is None:
        index = {}
        for code, entry in _g2p_registry().items():
            for key in [entry.get("name"), *(entry.get("alt_names") or [])]:
                if key:
                    index.setdefault(str(key).strip().strip('"').lower(), code)
        _NAME_INDEX = index
    return _NAME_INDEX


def _macrolanguage_candidates(code: str) -> list:
    """Codes worth trying when the exact one is absent from a registry.

    The registries disagree at the macrolanguage boundary: africa-g2p has `twi`
    but not `aka`, africa-corpus-builder has both. Trying the members of a
    macrolanguage (and vice versa) recovers most of these misses.
    """
    groups = {
        "aka": ["twi", "fat"],       # Akan -> Twi, Fanti
        "swa": ["swh", "swc"],       # Swahili
        "orm": ["gaz", "gax"],       # Oromo
        "ful": ["fuv", "fub", "ffm"],
        "mlg": ["plt"],              # Malagasy
        "kon": ["kng"],
        "lua": ["lua"],
    }
    out = list(groups.get(code, []))
    for macro, members in groups.items():
        if code in members and macro not in out:
            out.append(macro)
    return out


def resolve(token: str) -> Language:
    """Resolve a user-supplied language token into codes every stage can use."""
    token = token.strip()
    if not token:
        raise LanguageNotSupported("No language given.")

    code, name, family, countries = token.lower(), token, None, ()
    if _afriso is not None:
        try:
            record = _afriso.get(token)
        except Exception:
            record = None
        if record is not None:
            code = record.iso639_3
            name = record.name
            family = getattr(record, "family", None)
            countries = tuple(getattr(record, "countries", ()) or ())

    registry = _g2p_registry()
    g2p_code = None
    for candidate in [code, token.lower(), *_macrolanguage_candidates(code)]:
        if candidate in registry:
            g2p_code = candidate
            break
    if g2p_code is None:
        g2p_code = _g2p_name_index().get(token.lower()) or _g2p_name_index().get(code)

    # Without afriso, a name like "Yoruba" resolved to itself above; the g2p
    # registry knows the real code, so adopt it rather than passing a non-code on.
    lookup_code = code
    if g2p_code and code not in registry:
        code = registry[g2p_code].get("iso639_3") or g2p_code

    if g2p_code:
        entry = registry[g2p_code]
        # When a macrolanguage fell back to a member ("Akan" -> twi), the name
        # must follow the code: the run really is Twi, and the card should say so.
        fell_back = (entry.get("iso639_3") or g2p_code) != lookup_code
        if (name == token or fell_back) and entry.get("name"):
            name = entry["name"]
        family = family or entry.get("family")
        countries = countries or ((entry["country"],) if entry.get("country") else ())

    return Language(token=token, code=code, name=name, g2p_code=g2p_code,
                    family=family, countries=countries)


def require_g2p(language: Language) -> str:
    if not language.has_g2p:
        raise LanguageNotSupported(
            f"africa-g2p has no rule table for {language.name!r} ({language.code}). "
            f"Run with `normalise: none` to synthesise the raw text instead, or add a "
            f"table at https://github.com/AfriSpeech/africa-g2p"
        )
    return language.g2p_code


# BCP-47 language hints for Gemini Live transcription across African and common languages
_TRANSCRIBE_LANGUAGE_MAP: dict[str, list[str]] = {
    # Akan / Twi / Fante
    "aka": ["ak-GH", "ak", "tw"],
    "twi": ["ak-GH", "ak", "tw"],
    "fat": ["ak-GH", "ak"],
    # Yoruba
    "yor": ["yo-NG", "yo"],
    # Hausa
    "hau": ["ha-NG", "ha-NE", "ha"],
    # Igbo
    "ibo": ["ig-NG", "ig"],
    "igb": ["ig-NG", "ig"],
    # Swahili
    "swh": ["sw-KE", "sw-TZ", "sw"],
    "swa": ["sw-KE", "sw-TZ", "sw"],
    "swc": ["sw-CD", "sw"],
    # Amharic
    "amh": ["am-ET", "am"],
    # Zulu
    "zul": ["zu-ZA", "zu"],
    # Xhosa
    "xho": ["xh-ZA", "xh"],
    # Somali
    "som": ["so-SO", "so-KE", "so-ET", "so"],
    # Oromo
    "orm": ["om-ET", "om"],
    "gaz": ["om-ET", "om"],
    "gax": ["om-ET", "om"],
    # Lingala
    "lin": ["ln-CD", "ln-CG", "ln"],
    # Luganda
    "lug": ["lg-UG", "lg"],
    # Wolof
    "wol": ["wo-SN", "wo"],
    # Shona
    "sna": ["sn-ZW", "sn"],
    # Kinyarwanda
    "kin": ["rw-RW", "rw"],
    # Kirundi
    "run": ["rn-BI", "rn"],
    # Sesotho
    "sot": ["st-ZA", "st-LS", "st"],
    # Setswana
    "tsn": ["tn-ZA", "tn-BW", "tn"],
    # Xitsonga
    "tso": ["ts-ZA", "ts-MZ", "ts"],
    # Siswati
    "ssw": ["ss-ZA", "ss-SZ", "ss"],
    # Tshivenda
    "ven": ["ve-ZA", "ve"],
    # Southern Ndebele
    "nbl": ["nr-ZA", "nr"],
    # Northern Ndebele
    "nde": ["nd-ZW", "nd"],
    # Chichewa / Nyanja
    "nya": ["ny-MW", "ny-ZM", "ny"],
    # Bambara
    "bam": ["bm-ML", "bm"],
    # Fula / Fulfulde
    "ful": ["ff-SN", "ff-NG", "ff"],
    "fuv": ["ff-NG", "ff"],
    "fub": ["ff-CM", "ff"],
    "ffm": ["ff-ML", "ff"],
    # Ewe
    "ewe": ["ee-GH", "ee-TG", "ee"],
    # Fon
    "fon": ["fon-BJ", "fon"],
    # Ga
    "gaa": ["gaa-GH", "gaa"],
    # Dagbani
    "dag": ["dag-GH", "dag"],
    # Moore / Mossi
    "mos": ["mos-BF", "mos"],
    # Tigrinya
    "tir": ["ti-ET", "ti-ER", "ti"],
    # Malagasy
    "mlg": ["mg-MG", "mg"],
    "plt": ["mg-MG", "mg"],
    # Afrikaans
    "afr": ["af-ZA", "af"],
    # Arabic
    "ara": ["ar-EG", "ar"],
    # English
    "eng": ["en-GH", "en-NG", "en-ZA", "en-KE", "en-US", "en-GB", "en"],
    # Liberian English / Pidgin
    "lir": ["en-LR", "en"],
    # French
    "fra": ["fr-SN", "fr-CI", "fr-CD", "fr-CM", "fr-FR", "fr"],
    # Portuguese
    "por": ["pt-AO", "pt-MZ", "pt-PT", "pt"],
}


def resolve_transcribe_language_codes(language: Union[str, Language], limit: int = 4) -> List[str]:
    """Resolve a language into BCP-47 candidate codes for Live transcription.

    Gemini Live API's AudioTranscriptionConfig accepts BCP-47 language hints. Providing
    the regional codes (e.g. `ak-GH`, `yo-NG`) prevents phonetic confusion with other
    languages in multi-lingual models.
    """
    if isinstance(language, str):
        try:
            lang = resolve(language)
        except Exception:
            lang = Language(token=language, code=language.lower().strip(), name=language)
    else:
        lang = language

    codes: List[str] = []
    # Check table by ISO 639-3, original token, or g2p code
    for key in (lang.code, lang.token.lower(), getattr(lang, "g2p_code", None)):
        if key and key in _TRANSCRIBE_LANGUAGE_MAP:
            for c in _TRANSCRIBE_LANGUAGE_MAP[key]:
                if c not in codes:
                    codes.append(c)

    # Add country-qualified ISO codes where countries are known
    for country in getattr(lang, "countries", ()) or ():
        candidate = f"{lang.code}-{country}"
        if candidate not in codes:
            codes.append(candidate)

    # Always include the bare code if not already present
    if lang.code not in codes:
        codes.append(lang.code)

    return codes[:limit]
