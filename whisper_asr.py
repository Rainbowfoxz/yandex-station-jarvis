"""Точное распознавание вопроса после «Джарвиса»: faster-whisper офлайн на процессоре.

Модель small (int8, ~0,5 ГБ) скачивается при первом запуске в Downloads\\whisper.
"""
import pathlib

import numpy as np
from faster_whisper import WhisperModel

ROOT = pathlib.Path.home() / 'Downloads' / 'whisper'
SIZE = 'small'
# Подсказка словаря: имя и частые команды колонки, чтобы «моя волна» не превращалась в «моего ума».
HOTWORDS = 'Джарвис, Алиса, включи мою волну, Яндекс Музыка, плейлист, будильник, таймер, громкость, погода'
MAX_SECONDS = 8    # одна фраза; больше — склеивает с прошлой и дольше думает
BEAM = 1           # жадный поиск: 0,75 с вместо 0,96 с на фразу, на тестовых фразах текст тот же


class Whisper:
    def __init__(self):
        self.model = WhisperModel(SIZE, device='cpu', compute_type='int8', download_root=str(ROOT))

    def text(self, pcm):
        """PCM S16_LE 16 кГц моно -> текст ('' если речи нет)."""
        audio = np.frombuffer(pcm, '<i2').astype(np.float32)[-16000 * MAX_SECONDS:] / 32768
        segments, _ = self.model.transcribe(audio, language='ru', beam_size=BEAM, hotwords=HOTWORDS,
                                            condition_on_previous_text=False, vad_filter=True)
        return ' '.join(s.text.strip() for s in segments).strip()
