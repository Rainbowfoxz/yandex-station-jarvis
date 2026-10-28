"""Собрать кадры для /dev/fb0 из штатных анимаций .led (панель 84 LED + логотип 5 LED = 89 пикселей).

    py make_frames.py panel_thinking.led oknix_thinking.led thinking.bin
"""
import sys


def frames(path):
    out = []
    with open(path, encoding='ascii') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2 or not all(len(p) == 8 for p in parts[:-1]):
                continue  # служебные строки: loop, gammacorrection
            out.append([bytes.fromhex(p) for p in parts[:-1]])
    return out


panel, oknix = frames(sys.argv[1]), frames(sys.argv[2])
assert len(panel) == len(oknix), (len(panel), len(oknix))
data = b''
for p, o in zip(panel, oknix):
    pixels = p + o
    assert len(pixels) == 89, len(pixels)
    data += b''.join(pixels)
with open(sys.argv[3], 'wb') as f:
    f.write(data)
print(f'{len(panel)} кадров, {len(data)} байт')
