"""aec.c против aec.py на одной записи: music6.raw + «Джарвис» (mic_test.wav) с 10-й секунды, как test_aec.py.

    py test_aec_parity.py путь\\к\\aec_test.exe
Печатает расхождение C и Python (дБ ниже сигнала) и подавление музыки обеими версиями.
"""
import pathlib
import subprocess
import sys
import wave

import numpy as np
from scipy.signal import resample_poly

SETUP = pathlib.Path.home() / 'Downloads' / 'station-setup'
sys.path.insert(0, str(SETUP))
from aec import BLOCK, EchoCanceller  # noqa: E402

x = np.frombuffer((SETUP / 'music6.raw').read_bytes(), '<i2').reshape(-1, 6).astype(np.float64)
mic = resample_poly(x[:, 0], 1, 3)
refs = np.stack([resample_poly(x[:, 4], 1, 3), resample_poly(x[:, 5], 1, 3)])
with wave.open(str(SETUP / 'mic_test.wav')) as w:
    voice = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float64)
at = 16000 * 10
voice = voice[:len(mic) - at]
music_only = mic.copy()
mic[at:at + len(voice)] += voice
n = len(mic) // BLOCK * BLOCK


def as_int16(a):
    return np.clip(np.round(a), -32768, 32767).astype('<i2')


def run_c(m):
    data = np.stack([as_int16(m[:n]), as_int16(refs[0, :n]), as_int16(refs[1, :n])], axis=1).tobytes()
    out = subprocess.run([sys.argv[1]], input=data, capture_output=True, check=True).stdout
    return np.frombuffer(out, '<f4').astype(np.float64)


def run_py(m):
    aec = EchoCanceller(2)
    ints = np.stack([as_int16(refs[0, :n]), as_int16(refs[1, :n])]).astype(np.float64)
    return aec.process(as_int16(m[:n]).astype(np.float64), ints)


c, py = run_c(mic), run_py(mic)
diff = 10 * np.log10((py ** 2).mean() / ((c - py) ** 2).mean())
print(f'расхождение C и Python: {diff:.1f} дБ ниже сигнала')
seg = slice(16000 * 4, 16000 * 10)
for name, out in (('Python', run_py(music_only)), ('C', run_c(music_only))):
    print(f'подавление музыки ({name}): {10 * np.log10((music_only[seg] ** 2).mean() / (out[seg] ** 2).mean()):.1f} дБ')
ok = diff > 40
print('ИТОГ:', 'совпадает' if ok else 'РАСХОДИТСЯ')
sys.exit(0 if ok else 1)
