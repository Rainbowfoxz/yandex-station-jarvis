"""Проверка потока с микрофонов колонки: сколько байт пришло и уровень (RMS) по каналам.

    py mic_level.py [секунды]
"""
import subprocess
import sys
import threading
import time

import numpy as np

from jarvis import SSH

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 3
proc = subprocess.Popen(SSH + ['sh /data/opt/jarvis/pull.sh'], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
data = bytearray()
reader = threading.Thread(target=lambda: [data.extend(c) for c in iter(lambda: proc.stdout.read(14336), b'')],
                          daemon=True)
reader.start()
time.sleep(seconds)
proc.terminate()
got = bytes(data[:len(data) // 12 * 12])  # Алиса читает 6 каналов: кадр 12 байт
print(f'за {seconds:g} с пришло {len(got)} байт (норма ~{int(seconds * 48000 * 12)})')
if got:
    x = np.frombuffer(got, '<i2').reshape(-1, 6).astype(np.float64)
    print('RMS по каналам:', [int(v) for v in np.sqrt((x ** 2).mean(axis=0))])
    print('пик по каналам:', [int(v) for v in np.abs(x).max(axis=0)])
