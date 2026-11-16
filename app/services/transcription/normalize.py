"""Russian transcript normalization for TTS training.

Goals: make text pronounceable and consistent for the XTTS tokenizer without
changing what was said. Numbers become words, typographic punctuation is unified,
unsupported symbols are removed. Words in Latin script are kept but flagged, since
the model will read them with Russian-trained pronunciation.
"""
from __future__ import annotations

import re
import unicodedata

from num2words import num2words

_PUNCT_MAP = {
    "«": '"', "»": '"', "„": '"', "“": '"', "”": '"', "‟": '"',
    "‘": "'", "’": "'", "‚": "'", "`": "'",
    "—": " - ", "–": " - ", "−": "-", "‐": "-", "‑": "-",
    "…": "...", " ": " ", " ": " ", " ": " ",
}
_SYMBOLS = {
    "%": " процентов", "№": "номер ", "&": " и ", "+": " плюс ", "=": " равно ", "$": " долларов", "€": " евро",
    "₽": " рублей",
}
_ALLOWED = re.compile(r"[^0-9A-Za-zА-Яа-яЁё\s.,!?;:'\"()\-]")
_NUM = re.compile(r"(?<![\w])(\d+(?:[.,]\d+)?)(?![\w])")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
_CYR = re.compile(r"[А-Яа-яЁё]")


def _num_to_words(m: re.Match) -> str:
    s = m.group(1)
    try:
        if "," in s or "." in s:
            whole, frac = re.split(r"[.,]", s, maxsplit=1)
            return f"{num2words(int(whole), lang='ru')} целых {num2words(int(frac), lang='ru')}"
        return num2words(int(s), lang="ru")
    except (ValueError, OverflowError, NotImplementedError):
        return s


def normalize_ru(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "")
    for a, b in _PUNCT_MAP.items():
        t = t.replace(a, b)
    for a, b in _SYMBOLS.items():
        t = t.replace(a, b)
    # "1 000 000" -> "1000000"
    t = re.sub(r"(?<=\d)[  ](?=\d{3}\b)", "", t)
    t = _NUM.sub(_num_to_words, t)
    t = _ALLOWED.sub(" ", t)
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    t = re.sub(r"([.,!?;:])(?=[^\s.,!?;:\"')])", r"\1 ", t)
    t = re.sub(r"\s*-\s*-\s*", " - ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^[\s.,;:\-]+", "", t)
    return t


def text_warnings(text: str) -> list[str]:
    w = []
    if not text.strip():
        w.append("empty transcript")
        return w
    if _LATIN_WORD.search(text):
        w.append("contains Latin-script words")
    if not _CYR.search(text):
        w.append("no Cyrillic text")
    if re.search(r"\d", text):
        w.append("unexpanded digits")
    return w


def is_hallucination_like(text: str) -> bool:
    """Common Whisper hallucinations on music/silence in Russian YouTube audio."""
    low = text.lower()
    markers = ("субтитры", "редактор субтитров", "продолжение следует", "спасибо за просмотр",
               "подписывайтесь на канал", "dimatorzok", "amara.org")
    if any(m in low for m in markers):
        return True
    words = low.split()
    if len(words) >= 6 and len(set(words)) <= len(words) / 3:  # heavy repetition
        return True
    return False
