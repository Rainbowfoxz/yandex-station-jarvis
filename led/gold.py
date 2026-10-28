"""Перекрасить анимации Алисы (.led) в оттенки золота, сохранив яркость каждого пикселя.

    py gold.py        # orig/{panel,oknix}/alice_*.led -> gold/{panel,oknix}/
"""
import pathlib

HERE = pathlib.Path(__file__).parent
SHADES = {  # оттенок по действию, яркость берётся из оригинального пикселя
    # Светодиоды тянут зелёный: 255,200,40 на панели выглядит лимонным, поэтому G занижен.
    # Синий в ноль: любая примесь B на светодиодах выбеливает золото.
    'listening': (255, 115, 0),    # насыщенное золото
    'thinking': (255, 70, 0),      # тёмный янтарь
    'speaking': (255, 135, 0),     # светлое золото
}


def shade_for(name):
    for action, rgb in SHADES.items():
        if action in name:
            return rgb
    raise ValueError(name)


def recolor(token, rgb):
    r, g, b, w = bytes.fromhex(token)
    v = max(r, g, b)
    if v == 0:
        return token
    new = [round(v * c / 255) for c in rgb]
    return bytes(new + [w]).hex()


for src in sorted((HERE / 'orig').glob('*/alice_*.led')):
    rgb = shade_for(src.stem)
    out_lines = []
    for line in src.read_text(encoding='ascii').splitlines():
        parts = line.split()
        if len(parts) > 1 and all(len(p) == 8 for p in parts[:-1]):
            line = ' '.join([recolor(p, rgb) for p in parts[:-1]] + [parts[-1]])
        out_lines.append(line)
    dst = HERE / 'gold' / src.parent.name / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(('\n'.join(out_lines) + '\n').encode('ascii'))
    print(f'{src.parent.name}/{src.name}: {rgb}')
