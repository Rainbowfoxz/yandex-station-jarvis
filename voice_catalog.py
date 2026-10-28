"""Каталог записей голоса Джарвиса: текст (Whisper), длина и похожесть голоса на эталон jarvis-og.

Похожесть — косинус эмбеддингов ECAPA (SpeechBrain) с усреднённым эмбеддингом записей jarvis-og:
около 0,7 и выше — скорее тот же голос, ниже 0,5 — скорее чужой (музыка, другой персонаж, английский).
Пишет voice_catalog.tsv рядом с собой. Запускать питоном из окружения F5-TTS:

    %USERPROFILE%\\Downloads\\f5-tts\\venv\\Scripts\\python.exe voice_catalog.py <папка> [<папка> ...]
"""
import os
import pathlib
import sys

ROOT = pathlib.Path.home() / 'Downloads' / 'f5-tts'
os.environ.setdefault('HF_HOME', str(ROOT / 'hf'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402
from speechbrain.inference.speaker import EncoderClassifier  # noqa: E402
from speechbrain.utils.fetching import LocalStrategy  # noqa: E402

HERE = pathlib.Path(__file__).parent
REFERENCE = HERE / 'jarvis-og-wav'
OUT = HERE / 'voice_catalog.tsv'
RATE = 16000
EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}


def load(path):
    data, rate = sf.read(str(path), dtype='float32', always_2d=True)
    mono = data.mean(axis=1)
    n = int(len(mono) * RATE / rate)
    return np.interp(np.linspace(0, len(mono) - 1, n), np.arange(len(mono)), mono).astype(np.float32)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    whisper = WhisperModel('small', device=device, compute_type='float16' if device == 'cuda' else 'int8',
                           download_root=str(pathlib.Path.home() / 'Downloads' / 'whisper'))
    ecapa = EncoderClassifier.from_hparams('speechbrain/spkrec-ecapa-voxceleb',
                                           savedir=str(ROOT / 'ecapa'), run_opts={'device': device},
                                           local_strategy=LocalStrategy.COPY)   # симлинки без админа нельзя

    def embed(audio):
        with torch.no_grad():
            e = ecapa.encode_batch(torch.from_numpy(audio)[None].to(device)).squeeze().cpu().numpy()
        return e / np.linalg.norm(e)

    ref = np.mean([embed(load(p)) for p in sorted(REFERENCE.glob('*.wav'))], axis=0)
    ref /= np.linalg.norm(ref)

    rows = []
    for folder in sys.argv[1:] or [str(REFERENCE)]:
        for path in sorted(pathlib.Path(folder).rglob('*')):
            if path.suffix.lower() not in EXTS:
                continue
            audio = load(path)
            segments, info = whisper.transcribe(audio, language='ru', beam_size=5)
            text = ' '.join(s.text.strip() for s in segments)
            sim = float(embed(audio) @ ref)
            rows.append((sim, len(audio) / RATE, str(path), text))
            print(f'{sim:5.2f} {len(audio) / RATE:6.1f} с  {path.parent.name}/{path.name[:28]:28}  {text[:90]}', flush=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('similarity\tseconds\tpath\ttext\n')
        for sim, sec, path, text in rows:
            f.write(f'{sim:.3f}\t{sec:.2f}\t{path}\t{text}\n')
    print(f'Записей {len(rows)}, всего {sum(r[1] for r in rows) / 60:.1f} мин; каталог {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
