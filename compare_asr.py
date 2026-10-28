"""Сравнить Vosk и Whisper на записи с колонки: по каждой фразе — текст и время распознавания.

    py compare_asr.py phrases6.raw [--echo]   # --echo: вычитать эхо (не для фраз, сказанных самой колонкой)
"""
import json
import sys
import time

import numpy as np
import vosk

import jarvis_wake as jw
from whisper_asr import Whisper

raw = open(sys.argv[1], 'rb').read()
pcm = jw.MicDecoder(cancel_echo='--echo' in sys.argv).feed(raw)
vosk.SetLogLevel(-1)
full = vosk.KaldiRecognizer(vosk.Model(str(jw.MODEL)), 16000)
t = time.perf_counter()
whisper = Whisper()
print(f'Whisper загружен за {time.perf_counter() - t:.1f} с')
utter = bytearray()
for i in range(0, len(pcm), 3200):
    chunk = pcm[i:i + 3200]
    utter += chunk
    final = full.AcceptWaveform(chunk) or i + 3200 >= len(pcm)
    if not final:
        continue
    text = json.loads(full.Result() if i + 3200 < len(pcm) else full.FinalResult()).get('text', '')
    if text:
        t = time.perf_counter()
        heard = whisper.text(bytes(utter))
        print(f'Vosk:    {text!r}\nWhisper: {heard!r} ({time.perf_counter() - t:.1f} с, '
              f'{len(utter) / 32000:.1f} с звука)\n')
    utter.clear()
