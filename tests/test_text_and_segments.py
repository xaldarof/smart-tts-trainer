import numpy as np

from app.config.settings import ProcessingSettings
from app.services.audio.vad import intersect, silence_regions, speech_stats
from app.services.dataset import quality
from app.services.dataset.segmenter import AWord, assign_speakers, build_candidates
from app.services.inference.xtts_infer import chunk_text, validate_text
from app.services.transcription.normalize import is_hallucination_like, normalize_ru, text_warnings
from tests.conftest import tone


# ------------------------------------------------------------ normalization
def test_normalize_numbers_and_punct():
    assert normalize_ru("У меня 3 кота") == "У меня три кота"
    assert normalize_ru("«Привет» — сказал он…") == '"Привет" - сказал он...'
    assert "двенадцать целых пять" in normalize_ru("12,5")
    assert normalize_ru("1 000 000 рублей") == "один миллион рублей"
    assert normalize_ru("50%") == "пятьдесят процентов"
    assert normalize_ru("  привет ,мир  ") == "привет, мир"
    assert normalize_ru("a|b @ c") == "a b c"


def test_text_warnings_and_hallucinations():
    assert "contains Latin-script words" in text_warnings("Это Python код")
    assert "no Cyrillic text" in text_warnings("hello")
    assert text_warnings("") == ["empty transcript"]
    assert is_hallucination_like("Редактор субтитров А.Семкин")
    assert is_hallucination_like("да да да да да да да да")
    assert not is_hallucination_like("Сегодня мы поговорим о синтезе речи.")


# ----------------------------------------------------------------- VAD utils
def test_region_helpers():
    sp = [(1.0, 2.0), (3.0, 5.0)]
    assert silence_regions(sp, 6.0) == [(0.0, 1.0), (2.0, 3.0), (5.0, 6.0)]
    assert intersect(sp, [(1.5, 3.5)]) == [(1.5, 2.0), (3.0, 3.5)]
    assert speech_stats(sp, 6.0)["speech_sec"] == 3.0


# ---------------------------------------------------------------- segmenter
def _words(spec):
    """spec: list of (start, end, text, speaker)"""
    return [AWord(s, e, t, 0.95) for s, e, t, _ in spec], [(s, e, spk) for s, e, _, spk in spec]


def test_assign_speakers_and_uncertain():
    words, turns = _words([(0.0, 0.5, " Привет", "A"), (0.6, 1.0, " мир.", "A"), (1.5, 2.0, " Да", "B")])
    assign_speakers(words, turns, uncertain=[(1.4, 1.6)])
    assert [w.speaker for w in words] == ["A", "A", "B"]
    assert [w.uncertain for w in words] == [False, False, True]


def test_build_candidates_splits_on_speaker_and_pause():
    spec = [(0.0, 0.5, " Один", "A"), (0.6, 1.0, " два.", "A"), (1.1, 1.5, " Чужие", "B"), (1.6, 2.0, " слова.", "B"),
            (2.1, 2.5, " Снова", "A"), (2.6, 3.0, " я.", "A"), (5.0, 5.5, " После", "A"), (5.6, 6.0, " паузы.", "A")]
    words, turns = _words(spec)
    assign_speakers(words, turns, [])
    c = build_candidates(words, {"A"}, min_sec=0.1, max_sec=10, pause_sec=0.35)
    assert [x.text for x in c] == ["Один два.", "Снова я.", "После паузы."]
    # padding never crosses into neighbouring words
    assert c[0].end <= (1.0 + 1.1) / 2 + 1e-9 and c[1].start >= (2.0 + 2.1) / 2 - 1e-9


def test_build_candidates_respects_max_and_prefers_sentence_end():
    spec, t = [], 0.0
    for i in range(30):
        txt = f" слово{i}" + ("." if i % 6 == 5 else "")
        spec.append((t, t + 0.4, txt, "A"))
        t += 0.5
    words, turns = _words(spec)
    assign_speakers(words, turns, [])
    cands = build_candidates(words, {"A"}, min_sec=1.0, max_sec=5.0, pause_sec=1.0)
    assert all(c.words[-1].end - c.words[0].start <= 5.0 for c in cands)
    assert all(c.text.endswith(".") for c in cands[:-1])
    assert sum(len(c.words) for c in cands) == 30


# ------------------------------------------------------------------ quality
def test_quality_decisions():
    cfg = ProcessingSettings()
    speechlike = tone(4.0) * (np.sin(np.linspace(0, 20, int(4 * 22050))) > 0)   # on/off bursts -> high SNR estimate
    q = quality.assess("Нормальная фраза для обучения модели.", speechlike, 22050, 0.95, 0.9, 0.9, False, cfg)
    assert q.status == "accepted", q.reasons
    assert quality.assess("Коротко.", tone(0.5), 22050, 0.95, 0.9, 0.9, False, cfg).status == "rejected"
    assert quality.assess("Фраза.", speechlike, 22050, 0.4, 0.3, 0.9, False, cfg).status == "rejected"
    r = quality.assess("Нормальная фраза для обучения.", speechlike, 22050, 0.95, 0.9, 0.6, False, cfg)
    assert r.status == "review" and any("speaker identity" in x for x in r.reasons)
    assert quality.assess("Фраза для теста.", speechlike, 22050, 0.95, 0.9, 0.3, False, cfg).status == "rejected"
    assert quality.assess("Фраза для теста.", speechlike, 22050, 0.95, 0.9, 0.9, True, cfg).status == "rejected"
    clipped = np.clip(speechlike * 10, -1, 1)
    assert any("clipping" in x for x in quality.assess("Фраза для теста.", clipped, 22050, .95, .9, .9, False, cfg).reasons)
    r = quality.assess("Нормальная фраза для обучения.", speechlike, 22050, 0.95, 0.9, 0.9, False, cfg, ["numbers"])
    assert r.status == "review"


# ----------------------------------------------------------------- chunking
def test_chunk_text_limits():
    text = " ".join(["Это довольно длинное предложение, в котором много слов и запятых, чтобы проверить разбиение."] * 6)
    chunks = chunk_text(text, 170)
    assert all(len(c) <= 170 for c in chunks)
    assert " ".join(chunks).replace("  ", " ") == text
    one = chunk_text("а" * 400, 170)
    assert all(len(c) <= 171 for c in one) and "".join(one) == "а" * 400


def test_validate_text():
    import pytest

    from app.services.inference.xtts_infer import InferenceError

    assert validate_text("  Привет \n мир ") == "Привет мир"
    for bad in ("", "hello", "я" * 20001):
        with pytest.raises(InferenceError):
            validate_text(bad)
