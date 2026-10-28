"""Тест MicDecoder: запись с колонки, порезанная на куски случайной длины, должна дать «джарвис».

    py test_mic_decoder.py            # исправленный декодер
    py test_mic_decoder.py --old      # старая логика (хвост кадра отбрасывается) — должна провалиться
"""
import json
import random
import sys
import wave

import numpy as np
import vosk
from scipy.signal import resample_poly

import jarvis_wake as jw

random.seed(1)
with wave.open('mic_test.wav') as w:
    mono16 = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float64)
mono48 = resample_poly(mono16, 3, 1)
frames = np.zeros((len(mono48), jw.CHANNELS), '<i2')
frames[:, jw.MIC] = np.clip(mono48, -32768, 32767)
frames[:, 4] = 3000  # «чужие» каналы ненулевые, чтобы перепутанный канал был заметен
stream = frames.tobytes()

decoder = jw.MicDecoder()
if '--old' in sys.argv:
    def old_feed(block, d=decoder):
        d.rest = b''
        return jw.MicDecoder.feed(d, block[:len(block) // jw.FRAME * jw.FRAME])
    feed = old_feed
else:
    feed = decoder.feed

vosk.SetLogLevel(-1)
rec = vosk.KaldiRecognizer(vosk.Model(str(jw.MODEL)), 16000, jw.GRAMMAR)
texts, pos = [], 0
while pos < len(stream):
    n = random.randint(1000, 20000)
    if rec.AcceptWaveform(feed(stream[pos:pos + n])):
        texts.append(json.loads(rec.Result()).get('text', ''))
    pos += n
texts.append(json.loads(rec.FinalResult()).get('text', ''))
hits = sum('джарвис' in t for t in texts)
print('распознано:', [t for t in texts if t], '-> найдено «джарвис»:', hits)
sys.exit(0 if hits >= 2 else 1)
