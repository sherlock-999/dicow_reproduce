"""
NOTSOFAR adopts the same text normalizer as the CHiME-8 DASR track.
This code is aligned with the CHiME-8 repo:
https://github.com/chimechallenge/chime-utils/tree/main/chime_utils/text_norm
"""
import json
import os
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from .basic import BasicTextNormalizer as BasicTextNormalizer
from .english import EnglishTextNormalizer as EnglishTextNormalizerNSF


def get_text_norm(t_norm: str):
    if t_norm == 'whisper':
        SPELLING_CORRECTIONS = json.load(open(f'{os.path.dirname(__file__)}/english.json'))
        return EnglishTextNormalizer(SPELLING_CORRECTIONS)
    elif t_norm == 'whisper_nsf':
        return EnglishTextNormalizerNSF()
    elif t_norm == 'whisper_multi':
        return BasicTextNormalizer()
    elif t_norm == 'whisper_multi_char':
        # grapheme-level split for languages without word spaces (ja/th; CER convention)
        return BasicTextNormalizer(split_letters=True)
    else:
        return lambda x: x


# eval-time per-language normalizer dispatch: English keeps the
# NSF normalizer for continuity with every published English number; ja/ko/th
# are scored at character level (MLC-SLM CER convention); the rest use
# Whisper's multilingual basic normalizer.
CHAR_LEVEL_LANGUAGES = frozenset({"Japanese", "Korean", "Thai"})


def norm_name_for_language(language: str) -> str:
    base = (language or "").split("_")[0]
    if not base or base == "English":
        return 'whisper_nsf'
    if base in CHAR_LEVEL_LANGUAGES:
        return 'whisper_multi_char'
    return 'whisper_multi'