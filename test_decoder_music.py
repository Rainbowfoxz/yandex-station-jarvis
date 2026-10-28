"""Тест MicDecoder под музыкой: поток music6.raw + «Джарвис» в микрофоне, куски случайной длины.

    py test_decoder_music.py [--gain 0.5]    # с вычитанием эха — «джарвис» должен найтись
    py test_decoder_music.py --raw           # без вычитания — должен потеряться
"""
import json
import random
import sys
import wave

import numpy as np
import vosk
from scipy.signal import resample_poly

import jarvis_wake as jw

gain = float(sys.argv[sys.argv.index('--gain') + 1]) if '--gain' in sys.argv else 0.5
random.seed(2)
frames = np.frombuffer(open('music6.raw', 'rb').read(), '<i2').reshape(-1, jw.CHANNELS).astype(np.float64)
with wave.open('mic_test.wav') as w:
    voice = resample_poly(np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float64), 3, 1)
at = 48000 * 10
voice = voice[:len(frames) - at] * gain
frames[at:at + len(voice), jw.MIC] += voice
stream = np.clip(frames, -32768, 32767).astype('<i2').tobytes()

decoder = jw.MicDecoder(cancel_echo='--raw' not in sys.argv)
vosk.SetLogLevel(-1)
rec = vosk.KaldiRecognizer(vosk.Model(str(jw.MODEL)), 16000, jw.GRAMMAR)
texts, pos = [], 0
while pos < len(stream):
    n = random.randint(1000, 40000)
    if rec.AcceptWaveform(decoder.feed(stream[pos:pos + n])):
        texts.append(json.loads(rec.Result()).get('text', ''))
    pos += n
texts.append(json.loads(rec.FinalResult()).get('text', ''))
hits = sum('джарвис' in t for t in texts)
print('распознано:', [t for t in texts if t], '-> «джарвис»:', hits)
sys.exit(0 if hits else 1)
