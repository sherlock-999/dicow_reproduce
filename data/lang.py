"""Language handling for the multilingual (nemotron-3.5) backbone.

Maps MLC-SLM `language_or_accent` values (stored verbatim in supervision
.language, e.g. "English_American", "French") to the backbone's
prompt_dictionary locale keys. The backbone's supported tiers
(transcription-ready + broad-coverage, HF card) cover 10 of MLC's 11
languages; **Thai is adaptation-ready only and is excluded from training**
— it stays here solely so the eval diagnostic row can decode
it with an explicit th-TH prompt.

All English accents map to en-US: the 6 base cutsets carry no accent metadata,
so a single English prompt keeps conditioning uniform across corpora.

pt/es variant choice (pt-PT vs pt-BR, es-ES vs es-US) is finalized by the
gate-1 transcribe probe on mlc_dev clips; update here if the probe disagrees.
"""

TRAIN_LANGUAGES = frozenset({
    "English", "French", "German", "Italian", "Japanese", "Korean",
    "Portuguese", "Russian", "Spanish", "Vietnamese",
})

LANG_TO_LOCALE = {
    "English": "en-US",
    "French": "fr-FR",
    "German": "de-DE",
    "Italian": "it-IT",
    "Japanese": "ja-JP",
    "Korean": "ko-KR",
    "Portuguese": "pt-PT",  # CONFIRMED gate-1: MLC refs are European Portuguese
    "Russian": "ru-RU",
    "Spanish": "es-ES",     # CONFIRMED gate-1: MLC refs are peninsular Spanish
    "Vietnamese": "vi-VN",
    "Thai": "th-TH",        # eval diagnostic only (adaptation-ready tier)
}


def base_language(language_or_accent) -> str:
    """'English_American' -> 'English'; 'Thai' -> 'Thai'; None -> ''."""
    return (language_or_accent or "").split("_")[0]


def resolve_locale(language_or_accent, default: str = "en-US") -> str:
    """Supervision .language -> prompt_dictionary locale key.

    Missing/empty language (the 6 English base cutsets) -> `default`.
    A known base language uses LANG_TO_LOCALE; anything else raises — a silent
    en-US prompt on known-non-English audio would corrupt training.
    """
    base = base_language(language_or_accent)
    if not base:
        return default
    try:
        return LANG_TO_LOCALE[base]
    except KeyError:
        raise KeyError(f"No locale mapping for language {language_or_accent!r} "
                       f"(known: {sorted(LANG_TO_LOCALE)})") from None
