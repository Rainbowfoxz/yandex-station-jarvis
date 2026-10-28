"""Записать N секунд всех 6 каналов с колонки как есть (S16_LE, 48 кГц, кадр 12 байт).

    py mic_raw.py [секунды] [файл.raw]
"""
import subprocess
import sys
import threading
import time

from jarvis import SSH
from jarvis_wake import FRAME

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 10
out = sys.argv[2] if len(sys.argv) > 2 else 'mic6.raw'
proc = subprocess.Popen(SSH + ['sh /data/opt/jarvis/pull.sh'], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
data = bytearray()
threading.Thread(target=lambda: [data.extend(c) for c in iter(lambda: proc.stdout.read(12288), b'')],
                 daemon=True).start()
time.sleep(seconds)
proc.terminate()
raw = bytes(data[:len(data) // FRAME * FRAME])
with open(out, 'wb') as f:
    f.write(raw)
print(f'{out}: {len(raw) / FRAME / 48000:.1f} с')
