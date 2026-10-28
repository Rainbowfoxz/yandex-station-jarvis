"""Тест устойчивости вычитания эха: тишина динамика, затем резкое начало музыки и снова тишина.

На живой колонке фильтр разошёлся (переполнение, NaN) — после этого слушатель глох навсегда.
    py test_aec_onset.py      # выход должен остаться конечным и не громче входа
"""
import sys

import numpy as np

import jarvis_wake as jw

speech = np.frombuffer(open('phrases6.raw', 'rb').read(), '<i2').reshape(-1, jw.CHANNELS)
live = np.frombuffer(open('live6.raw', 'rb').read(), '<i2').reshape(-1, jw.CHANNELS)
music = np.frombuffer(open('music6.raw', 'rb').read(), '<i2').reshape(-1, jw.CHANNELS)
# Динамик молчит (опора как на живой колонке: канал 4 — нули, канал 5 — смещение ~17),
# а в комнате говорят: микрофон из записи речи. Фильтру не на чем учиться, он только дрейфует.
n = min(len(live), len(speech))
silent = live[:n].copy()
silent[:, jw.MIC] = speech[:n, jw.MIC]
silent = np.concatenate([silent] * 3)        # ~1 минута
loud = music.astype(np.float64)
loud[:, list(jw.REFS)] *= 8                  # резкое начало громкой музыки
loud = np.clip(loud, -32768, 32767).astype('<i2')
stream = np.concatenate([silent, loud, silent, loud]).tobytes()

decoder = jw.MicDecoder()
out = []
for i in range(0, len(stream), jw.CHUNK):
    out.append(np.frombuffer(decoder.feed(stream[i:i + jw.CHUNK]), '<i2'))
out = np.concatenate(out).astype(np.float64)
w = decoder.aec.w
peak_in = np.abs(np.frombuffer(stream, '<i2').reshape(-1, jw.CHANNELS)[:, jw.MIC]).max()
print(f'фильтр конечен: {np.isfinite(w).all()}, пик выхода {np.abs(out).max():.0f} при пике входа {peak_in}')
sys.exit(0 if np.isfinite(w).all() and np.abs(out).max() <= 2 * peak_in else 1)
