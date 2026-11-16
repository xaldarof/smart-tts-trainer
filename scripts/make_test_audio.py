"""Build a synthetic two-speaker Russian test recording (macOS only: uses `say -v Milena`).

Speaker A = Milena voice; speaker B = Milena pitch-shifted down ~5 semitones (distinct
embedding). Adds a quiet synthetic music bed so the enhancement stage has something to
remove. Used for local end-to-end verification without downloading anyone's voice.

    python scripts/make_test_audio.py out.wav [--minutes-a 3]
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SENTENCES_A = [
    "Сегодня мы поговорим о том, как устроены современные системы синтеза речи.",
    "Первым делом нужно собрать качественные записи голоса без посторонних шумов.",
    "Каждый фрагмент должен содержать одну законченную фразу.",
    "Длина фрагментов обычно составляет от трёх до десяти секунд.",
    "Затем записи автоматически расшифровываются с помощью распознавания речи.",
    "После этого текст проверяется и исправляется вручную.",
    "Модель учится сопоставлять буквы и звуки конкретного человека.",
    "Чем больше чистых данных, тем естественнее звучит результат.",
    "Важно получить согласие человека, чей голос используется для обучения.",
    "Видеокарта с шестнадцатью гигабайтами памяти справляется с такой задачей.",
    "Обучение занимает от нескольких часов до одного дня.",
    "Контрольные точки позволяют продолжить обучение после перерыва.",
    "Синтезированную речь можно сохранить в формате вав или эм пэ три.",
    "Длинный текст разбивается на предложения и озвучивается по частям.",
    "Хорошая интонация зависит от знаков препинания в исходном тексте.",
    "Мы сравним дообученную модель с исходной и послушаем разницу.",
    "Шум, музыка и эхо заметно ухудшают качество обучающих данных.",
    "Поэтому фоновые звуки сначала отделяются от голоса.",
    "Затем определяются участки, где говорит каждый человек.",
    "Пользователь сам выбирает, чей голос попадёт в набор данных.",
    "Сомнительные фрагменты отправляются на ручную проверку.",
    "Повторяющиеся фразы, например приветствия, удаляются автоматически.",
    "Небольшая часть данных откладывается для проверки качества.",
    "Во время обучения отображается функция потерь и использование памяти.",
    "Если памяти не хватает, размер пакета автоматически уменьшается.",
    "После завершения модель сохраняется в локальной библиотеке.",
    "Теперь можно ввести любой русский текст и прослушать результат.",
    "Спасибо, что дослушали этот небольшой рассказ до конца.",
]
SENTENCES_B = [
    "А как быстро всё это работает на обычном компьютере?",
    "Интересно, а нужен ли для этого интернет?",
    "Понятно. А сколько записей нужно для хорошего результата?",
    "Звучит разумно, давайте попробуем.",
    "А можно ли использовать чужой голос без разрешения?",
    "Хорошо, спасибо за подробный ответ.",
]


def tts(text: str, sr: int, rate: int = 175) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        aiff = Path(td) / "x.aiff"
        subprocess.run(["say", "-v", "Milena", "-r", str(rate), "-o", str(aiff), text], check=True)
        x, s = sf.read(str(aiff), dtype="float32", always_2d=True)
    return soxr.resample(x.mean(1), s, sr).astype(np.float32)


def pitch_down(x: np.ndarray, sr: int, factor: float = 0.75) -> np.ndarray:
    # resample-based pitch shift (also slows down): play at lower rate
    return soxr.resample(x, sr, int(sr / factor)).astype(np.float32)


def music(n: int, sr: int, level_db: float = -30) -> np.ndarray:
    t = np.arange(n) / sr
    chords = [(220, 277.18, 329.63), (196, 246.94, 293.66), (174.61, 220, 261.63)]
    out = np.zeros(n, dtype=np.float32)
    seg = 2 * sr
    for i in range(0, n, seg):
        f = chords[(i // seg) % len(chords)]
        tt = t[i:i + seg]
        out[i:i + seg] = sum(np.sin(2 * np.pi * fr * tt) for fr in f) / 3
    return out * (10 ** (level_db / 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--repeat", type=int, default=2, help="repeat speaker-A sentence list N times (varied rate)")
    a = ap.parse_args()
    sr = 44100
    parts = []
    b_iter = iter(SENTENCES_B * 10)
    rates = [165, 185, 175, 195]
    for rep in range(a.repeat):
        for i, s in enumerate(SENTENCES_A):
            parts.append(tts(s, sr, rates[(i + rep) % len(rates)]))
            parts.append(np.zeros(int(0.6 * sr), dtype=np.float32))
            if i % 5 == 4:
                parts.append(pitch_down(tts(next(b_iter), sr), sr))
                parts.append(np.zeros(int(0.7 * sr), dtype=np.float32))
    x = np.concatenate([np.zeros(sr, dtype=np.float32), *parts])
    x = 0.8 * x / np.max(np.abs(x)) + music(len(x), sr)
    sf.write(a.out, x, sr, subtype="PCM_16")
    print(f"wrote {a.out}: {len(x) / sr:.1f}s")


if __name__ == "__main__":
    main()
