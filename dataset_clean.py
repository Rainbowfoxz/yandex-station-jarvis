"""Очистка реплик «Джарвис, но музыка/шум» (отметка 2 в jarvis_review.json): Demucs оставляет только голос.

Реплику из фильма режем стерео 44,1 кГц с запасом CONTEXT по краям (разделителю нужен контекст),
после разделения обрезаем запас и сводим в моно 24 кГц. Результат — Downloads\\f5-tts\\review2
со страницей для повторного прослушивания (ссылка «до очистки» — исходный фрагмент).
Запускать питоном из окружения F5-TTS:

    ... dataset_clean.py
"""
import json
import os
import pathlib
import subprocess

F5 = pathlib.Path.home() / 'Downloads' / 'f5-tts'
os.environ.setdefault('TORCH_HOME', str(F5 / 'torch'))   # веса Demucs — в Загрузки, не в кэш на C

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from demucs.api import Separator  # noqa: E402

HERE = pathlib.Path(__file__).parent
REVIEW = F5 / 'review'
OUT = F5 / 'review2'
FILMS = pathlib.Path(r'D:\code\alice')
CONTEXT = 1.0       # с, запас звука вокруг реплики для разделителя
RATE = 24000
PAD = 0.1, 0.2      # как в dataset_review.py


def film_audio(item, rate):
    """Стерео фрагмент фильма с запасом -> (тензор [2, T], сколько секунд запаса слева)."""
    a = max(item['start'] - PAD[0] - CONTEXT, 0)
    left = item['start'] - PAD[0] - a
    dur = item['end'] + PAD[1] + CONTEXT - a
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{a:.2f}', '-t', f'{dur:.2f}',
                          '-i', str(FILMS / f'{item["src"]}.mp4'), '-map', '0:a:0', '-ac', '2', '-ar', str(rate),
                          '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    return torch.from_numpy(np.frombuffer(raw, '<f4').reshape(-1, 2).T.copy()), left


def main():
    items = json.loads((pathlib.Path.home() / 'Downloads' / 'jarvis_review.json').read_text(encoding='utf-8'))
    noisy = [i for i in items if i['verdict'] == 2]
    sep = Separator(model='htdemucs_ft', device='cuda', progress=False)
    rate = sep.samplerate
    (OUT / 'clips').mkdir(parents=True, exist_ok=True)
    out = []
    for n, item in enumerate(noisy, 1):
        if 'start' in item:
            wav, left = film_audio(item, rate)
            keep = item['end'] + PAD[1] - (item['start'] - PAD[0])
        else:   # запись пака — целиком, без запаса
            data, sr = sf.read(str(REVIEW / item['file']), dtype='float32')
            wav, left, keep = torch.from_numpy(np.stack([data, data])), 0.0, len(data) / sr
            if sr != rate:
                wav = torch.from_numpy(np.stack([np.interp(np.arange(0, len(data), sr / rate),
                                                           np.arange(len(data)), data).astype(np.float32)] * 2))
        _, stems = sep.separate_tensor(wav, rate)
        voice = stems['vocals'].mean(0).cpu().numpy()[int(left * rate):int((left + keep) * rate)]
        voice = np.interp(np.arange(0, len(voice), rate / RATE), np.arange(len(voice)), voice).astype(np.float32)
        name = f'{n:03}.wav'
        sf.write(str(OUT / 'clips' / name), voice, RATE, subtype='PCM_16')
        out.append(dict(item, review_id=item['id'], id=n, file=f'clips/{name}', orig=f'../review/{item["file"]}',
                        verdict=0))
        print(f'{n}/{len(noisy)} {item["text"][:60]}', flush=True)
    page = ((HERE / 'dataset_review.html').read_text(encoding='utf-8')
            .replace('/*ITEMS*/[]', json.dumps(out, ensure_ascii=False))
            .replace("/*SAVE_AS*/'jarvis_review.json'", "'jarvis_review2.json'")
            .replace('<h1>Отбор реплик Джарвиса</h1>', '<h1>Реплики после очистки</h1>'))
    (OUT / 'index.html').write_text(page, encoding='utf-8')
    print(f'Готово: {OUT / "index.html"}')


if __name__ == '__main__':
    main()
