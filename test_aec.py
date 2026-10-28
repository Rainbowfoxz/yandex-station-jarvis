"""Тест вычитания эха на записи с музыкой (music6.raw) и подмешанным «Джарвис» (mic_test.wav).

    py test_aec.py            # с вычитанием эха
    py test_aec.py --off      # без вычитания — «Джарвис» под музыкой должен теряться
Печатает подавление музыки (дБ) и что услышала грамматика.
"""
import json
import sys
import wave

import numpy as np
import vosk
from scipy.signal import resample_poly

from aec import EchoCanceller
from jarvis_wake import GRAMMAR, MODEL

VOICE_GAIN = float(sys.argv[sys.argv.index('--gain') + 1]) if '--gain' in sys.argv else 1.0

x = np.frombuffer(open('music6.raw', 'rb').read(), '<i2').reshape(-1, 6).astype(np.float64)
mic = resample_poly(x[:, 0], 1, 3)
refs = np.stack([resample_poly(x[:, 4], 1, 3), resample_poly(x[:, 5], 1, 3)])

with wave.open('mic_test.wav') as w:
    voice = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float64)
# Голос «человека» поверх громкой музыки (с 10-й секунды), опорные каналы его не содержат.
at = 16000 * 10
voice = voice[:len(mic) - at] * VOICE_GAIN
music_only = mic.copy()
mic[at:at + len(voice)] += voice

if '--off' in sys.argv:
    clean = mic
else:
    clean = np.zeros(len(mic))
    aec = EchoCanceller(2)
    pos = 0
    for start in range(0, len(mic), 4096):                     # кусками, как в живом потоке
        out = aec.process(mic[start:start + 4096], refs[:, start:start + 4096])
        clean[pos:pos + len(out)] = out
        pos += len(out)
    # Подавление считаем на музыке без голоса: прогоняем тот же фильтр по чистой музыке.
    probe, aec2, pos = np.zeros(len(mic)), EchoCanceller(2), 0
    for start in range(0, len(mic), 4096):
        out = aec2.process(music_only[start:start + 4096], refs[:, start:start + 4096])
        probe[pos:pos + len(out)] = out
        pos += len(out)
    seg = slice(16000 * 4, 16000 * 10)
    print(f'подавление музыки: {10 * np.log10((music_only[seg] ** 2).mean() / (probe[seg] ** 2).mean()):.1f} дБ')

voice_rms = np.sqrt((voice ** 2).mean())
music_rms = np.sqrt((music_only[at:] ** 2).mean())
print(f'голос/музыка до обработки: {20 * np.log10(voice_rms / music_rms):.1f} дБ')

vosk.SetLogLevel(-1)
rec = vosk.KaldiRecognizer(vosk.Model(str(MODEL)), 16000, GRAMMAR)
pcm = np.clip(clean, -32768, 32767).astype('<i2').tobytes()
texts = []
for i in range(0, len(pcm), 3200):
    if rec.AcceptWaveform(pcm[i:i + 3200]):
        texts.append(json.loads(rec.Result())['text'])
texts.append(json.loads(rec.FinalResult())['text'])
hits = sum('джарвис' in t for t in texts)
print('услышано:', [t for t in texts if t], '-> «джарвис»:', hits)
sys.exit(0 if hits else 1)
