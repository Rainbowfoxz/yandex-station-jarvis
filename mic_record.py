"""Записать N секунд с микрофона колонки (канал 0, 16 кГц) в WAV и показать, что слышит Vosk.

    py mic_record.py [секунды] [файл.wav]
"""
import json
import subprocess
import sys
import threading
import time
import wave

import numpy as np
import vosk
from scipy.signal import resample_poly

from jarvis import SSH
from jarvis_wake import CHANNELS, FRAME, GRAMMAR, MIC, MODEL

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 6
out = sys.argv[2] if len(sys.argv) > 2 else 'mic.wav'
proc = subprocess.Popen(SSH + ['sh /data/opt/jarvis/pull.sh'], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
data = bytearray()
threading.Thread(target=lambda: [data.extend(c) for c in iter(lambda: proc.stdout.read(12288), b'')],
                 daemon=True).start()
time.sleep(seconds)
proc.terminate()
raw = bytes(data[:len(data) // FRAME * FRAME])
mic = np.frombuffer(raw, '<i2').reshape(-1, CHANNELS)[:, MIC].astype(np.float64)
pcm = np.clip(resample_poly(mic, 1, 3), -32768, 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(pcm.tobytes())
print(f'{out}: {len(pcm) / 16000:.1f} с, пик {int(np.abs(pcm).max())}, RMS {np.sqrt((pcm.astype(float) ** 2).mean()):.0f}')
vosk.SetLogLevel(-1)
model = vosk.Model(str(MODEL))
for grammar in (GRAMMAR, None):
    rec = vosk.KaldiRecognizer(model, 16000, grammar) if grammar else vosk.KaldiRecognizer(model, 16000)
    texts = []
    b = pcm.tobytes()
    for i in range(0, len(b), 3200):
        if rec.AcceptWaveform(b[i:i + 3200]):
            texts.append(json.loads(rec.Result()).get('text', ''))
    texts.append(json.loads(rec.FinalResult()).get('text', ''))
    print('грамматика' if grammar else 'свободно  ', [t for t in texts if t])
