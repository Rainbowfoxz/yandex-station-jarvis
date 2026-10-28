"""Собрать образцы голоса для jarvis_voice.py из каталога voice_catalog.tsv в Downloads\\f5-tts\\ref.

Имена файлов в папке пользователя испорчены (кириллица потеряна), поэтому записи ищем по тексту
из каталога. Рассказ про Ванко режем по меткам слов Whisper — нужно только первое предложение.
Запускать питоном из окружения F5-TTS.
"""
import csv
import os
import pathlib
import sys

ROOT = pathlib.Path.home() / 'Downloads' / 'f5-tts'
os.environ.setdefault('HF_HOME', str(ROOT / 'hf'))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

HERE = pathlib.Path(__file__).parent
REF_DIR = ROOT / 'ref'
# Имя файла -> начало текста в каталоге (только записи из папки пользователя, похожесть >= 0,6).
PICK = {
    'film_device': 'К сожалению, устройство',
    'film_suit': 'Судя по всему',
    'film_visual': 'Создать визуальный образ',
    'film_vanko_full': 'Антон Ванка',
}
STORY_END = 'депортирован'   # конец первого предложения рассказа


def main():
    REF_DIR.mkdir(parents=True, exist_ok=True)
    with open(HERE / 'voice_catalog.tsv', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f, delimiter='\t') if 'jarvis-voice' in r['path']]
    for name, start in PICK.items():
        row = max((r for r in rows if r['text'].startswith(start)), key=lambda r: float(r['similarity']))
        data, rate = sf.read(row['path'], dtype='float32', always_2d=True)
        sf.write(str(REF_DIR / f'{name}.wav'), data.mean(axis=1), rate)
        print(f'{name}: {float(row["seconds"]):.1f} с, похожесть {row["similarity"]}')

    data, rate = sf.read(str(REF_DIR / 'film_vanko_full.wav'), dtype='float32')
    audio16 = np.interp(np.arange(0, len(data), rate / 16000), np.arange(len(data)), data).astype(np.float32)
    # Без импорта torch на cuda не находится cublas64_12.dll, а для одного файла хватит процессора.
    whisper = WhisperModel('small', device='cpu', compute_type='int8',
                           download_root=str(pathlib.Path.home() / 'Downloads' / 'whisper'))
    segments, _ = whisper.transcribe(audio16, language='ru', word_timestamps=True)
    words = [w for s in segments for w in s.words]
    end = next(w.end for w in words if STORY_END in w.word)
    nxt = next((w.start for w in words if w.start > end), end + 0.3)
    cut = (end + nxt) / 2                      # середина паузы после слова
    sf.write(str(REF_DIR / 'film_vanko.wav'), data[:int(cut * rate)], rate)
    print(f'film_vanko: первое предложение до {cut:.2f} с: '
          + ''.join(w.word for w in words if w.end <= end + 0.01).strip())
    return 0


if __name__ == '__main__':
    sys.exit(main())
