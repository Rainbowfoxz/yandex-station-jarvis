"""Найти реплики Джарвиса в фильме: Whisper размечает речь, ECAPA сравнивает голос с эталоном.

Эталон — среднее эмбеддингов записей jarvis-og и реплик из voice_catalog.tsv с похожестью >= 0,55
(одна и та же озвучка дубляжа). Звук дорожки и список реплик — в Downloads\\f5-tts\\films.
Запускать питоном из окружения F5-TTS:

    ... film_jarvis.py D:\\code\\alice\\zheleznyi-chelovek.mp4 --track 0          # весь фильм
    ... film_jarvis.py <фильм> --track 2 --start 2700 --dur 600                   # кусок — выбрать дорожку
"""
import argparse
import csv
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path.home() / 'Downloads' / 'f5-tts'
os.environ.setdefault('HF_HOME', str(ROOT / 'hf'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402  — до faster_whisper: подгружает DLL CUDA
from faster_whisper import WhisperModel  # noqa: E402
from speechbrain.inference.speaker import EncoderClassifier  # noqa: E402
from speechbrain.utils.fetching import LocalStrategy  # noqa: E402

HERE = pathlib.Path(__file__).parent
FILMS = ROOT / 'films'
RATE = 16000
MIN_SEC, MAX_SEC = 1.0, 15.0     # короче — эмбеддинг ненадёжен, длиннее — для обучения не нужно
REF_SIM = 0.55                   # реплики из каталога, которые считаем эталоном
CHUNK = 600                      # секунд звука за один проход Whisper


def extract(film, track, start, dur, rate, out):
    """Дорожка фильма -> WAV моно нужной частоты (ffmpeg)."""
    cut = (['-ss', str(start)] if start else []) + (['-t', str(dur)] if dur else [])
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', *cut, '-i', str(film), '-map', f'0:a:{track}',
                    '-ac', '1', '-ar', str(rate), '-c:a', 'pcm_s16le', str(out)], check=True)


def reference(embed):
    paths = sorted((HERE / 'jarvis-og-wav').glob('*.wav'))
    with open(HERE / 'voice_catalog.tsv', encoding='utf-8') as f:
        paths += [pathlib.Path(r['path']) for r in csv.DictReader(f, delimiter='\t')
                  if 'jarvis-voice' in r['path'] and float(r['similarity']) >= REF_SIM
                  and 1.0 <= float(r['seconds']) <= 20]
    vectors = []
    for path in paths:
        data, rate = sf.read(str(path), dtype='float32', always_2d=True)
        mono = data.mean(axis=1)
        mono = np.interp(np.arange(0, len(mono), rate / RATE), np.arange(len(mono)), mono).astype(np.float32)
        vectors.append(embed(mono))
    ref = np.mean(vectors, axis=0)
    return ref / np.linalg.norm(ref), len(paths)


def main():
    parser = argparse.ArgumentParser(description='Реплики Джарвиса в фильме.')
    parser.add_argument('film')
    parser.add_argument('--track', type=int, default=0, help='Номер звуковой дорожки (0 — первая)')
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--dur', type=float, default=0, help='Сколько секунд обработать (0 — всё)')
    args = parser.parse_args()
    film = pathlib.Path(args.film)
    FILMS.mkdir(parents=True, exist_ok=True)
    tag = f'{film.stem}_t{args.track}' + (f'_{int(args.start)}+{int(args.dur)}' if args.dur else '')
    wav16 = FILMS / f'{tag}_16k.wav'

    t = time.perf_counter()
    extract(film, args.track, args.start, args.dur, RATE, wav16)
    audio, _ = sf.read(str(wav16), dtype='float32')
    print(f'Звук: {len(audio) / RATE / 60:.1f} мин за {time.perf_counter() - t:.0f} с', flush=True)

    whisper = WhisperModel('small', device='cuda', compute_type='float16',
                           download_root=str(pathlib.Path.home() / 'Downloads' / 'whisper'))
    ecapa = EncoderClassifier.from_hparams('speechbrain/spkrec-ecapa-voxceleb', savedir=str(ROOT / 'ecapa'),
                                           run_opts={'device': 'cuda'}, local_strategy=LocalStrategy.COPY)

    def embed(x):
        with torch.no_grad():
            e = ecapa.encode_batch(torch.from_numpy(x)[None].cuda()).squeeze().cpu().numpy()
        return e / np.linalg.norm(e)

    ref, n_ref = reference(embed)
    t = time.perf_counter()
    rows = []
    # Кусками: целый фильм в Whisper не влезает в память. Кусок захватывает MAX_SEC следующего,
    # чтобы фраза на стыке не порвалась; берём только фразы, начавшиеся внутри своего куска.
    for begin in range(0, len(audio), CHUNK * RATE):
        piece = audio[begin:begin + (CHUNK + int(MAX_SEC)) * RATE]
        segments, _ = whisper.transcribe(piece, language='ru', beam_size=5, vad_filter=True)
        for s in segments:
            if s.start >= CHUNK or not MIN_SEC <= s.end - s.start <= MAX_SEC:
                continue
            a, b = begin / RATE + s.start, begin / RATE + s.end
            sim = float(embed(audio[int(a * RATE):int(b * RATE)]) @ ref)
            rows.append((sim, args.start + a, args.start + b, s.text.strip()))
        print(f'  {(begin / RATE + CHUNK) / 60:.0f} мин: реплик {len(rows)}', flush=True)
    print(f'Реплик {len(rows)}, разметка и сравнение за {time.perf_counter() - t:.0f} с (эталон из {n_ref})',
          flush=True)

    out = FILMS / f'{tag}.tsv'
    with open(out, 'w', encoding='utf-8') as f:
        f.write('similarity\tstart\tend\ttext\n')
        for sim, a, b, text in sorted(rows, key=lambda r: r[1]):
            f.write(f'{sim:.3f}\t{a:.2f}\t{b:.2f}\t{text}\n')
    for limit in (0.6, 0.5, 0.45, 0.4):
        picked = [r for r in rows if r[0] >= limit]
        print(f'  похожесть >= {limit}: {len(picked):3} реплик, {sum(r[2] - r[1] for r in picked) / 60:.1f} мин')
    print('Самые похожие:')
    for sim, a, b, text in sorted(rows, reverse=True)[:8]:
        print(f'  {sim:.2f} {int(a) // 60:3}:{a % 60:04.1f}  {text[:80]}')
    print(f'Список: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
