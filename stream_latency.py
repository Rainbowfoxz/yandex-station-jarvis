"""Замер задержки потока микрофона: колонка проигрывает клип, ждём его в опорном канале на ПК.

    py stream_latency.py [повторов]
Печатает, через сколько после команды звук динамика появился в потоке (включая запуск gst ~0,1–0,3 с).
"""
import subprocess
import sys
import threading
import time

import numpy as np

from jarvis import SSH
from jarvis_listen import GST
from jarvis_wake import CHANNELS, FRAME, REFS

runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
proc = subprocess.Popen(SSH + ['sh /data/opt/jarvis/pull.sh'], stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE, bufsize=0)
shell = subprocess.Popen(SSH + ['sh'], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
arrivals = []          # (время прихода на ПК, был ли звук в опоре)
lock = threading.Lock()


def reader():
    rest = b''
    while True:
        block = proc.stdout.read(65536)
        if not block:
            return
        now = time.monotonic()
        block = rest + block
        cut = len(block) // FRAME * FRAME
        frames, rest = block[:cut], block[cut:]
        ref = np.frombuffer(frames, '<i2').reshape(-1, CHANNELS)[:, REFS[0]].astype(np.float64)
        with lock:
            arrivals.append((now, np.abs(ref).max() > 200, len(frames)))


threading.Thread(target=reader, daemon=True).start()
time.sleep(3)
for _ in range(runs):
    with lock:
        arrivals.clear()
    t0 = time.monotonic()
    shell.stdin.write((GST.format(name='reply1') + ' >/dev/null 2>&1 &\n').encode())
    shell.stdin.flush()
    found = None
    while time.monotonic() - t0 < 10 and found is None:
        time.sleep(0.02)
        with lock:
            found = next((t for t, loud, _ in arrivals if loud), None)
    with lock:
        sizes = sorted({n for _, _, n in arrivals})
    print(f'звук в потоке через {found - t0:.2f} с' if found else 'звук не пришёл за 10 с',
          f'(размеры кусков: {sizes[:5]})', flush=True)
    time.sleep(4)
proc.terminate()
shell.terminate()
