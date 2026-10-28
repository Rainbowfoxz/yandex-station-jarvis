"""Сравнение контрольных точек дообучения с исходной ESpeech на новых фразах (их нет в датасете).

Для каждой точки: похожесть голоса (ECAPA, эталон — чистые записи пака), разборчивость (доля ошибок
в буквах, когда Whisper large-v3 распознаёт синтез обратно) и длительность. WAV — в f5-tts\\out\\ft\\<точка>.
Запускать питоном из окружения F5-TTS, когда обучение закончено (видеокарта свободна):

    ... jarvis_compare.py                                    # все model_*.pt последнего прогона
    ... jarvis_compare.py jarvis/model_2000.pt jarvis2/model_500.pt   # выбранные (прогон/точка)
"""
import csv
import difflib
import pathlib
import re
import sys

import jarvis_voice as jv
from jarvis_voice import DiT, MODEL_CFG, MODEL_REPO, ROOT, hf_hub_download, load_model, np, sf, torch

CKPTS = ROOT / 'venv' / 'Lib' / 'ckpts'
OUT = ROOT / 'out' / 'ft'
PHRASES = [
    'Добрый вечер сэр, все системы работают в штатном режиме.',
    'Сейчас в Екатеринбурге плюс три градуса, к вечеру ожидается мокрый снег.',
    'Позвольте заметить, что кофеин не заменяет сон, сэр.',
    'Напоминаю, что встреча с директором назначена на четверг.',
    'Музыку включила Алиса, громкость на сорок процентов.',
]


def letters(text):
    return re.sub(r'[^а-я]', '', text.lower().replace('ё', 'е'))


def cer(ref, hyp):
    """Доля ошибок в буквах (1 - доля совпавших по выравниванию)."""
    a, b = letters(ref), letters(hyp)
    matched = sum(m.size for m in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return 1 - matched / max(len(a), len(b), 1)


def main():
    from faster_whisper import WhisperModel
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    run = max((p for p in CKPTS.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    names = sys.argv[1:] or [f'{run.name}/{p.name}' for p in sorted(
        (p for p in run.glob('model_*.pt') if p.name != 'model_last.pt'), key=lambda p: int(re.sub(r'\D', '', p.name)))]
    voice = jv.Voice(ckpt=None)   # «исходная» — ESpeech без дообучения
    whisper = WhisperModel('large-v3', device='cuda', compute_type='float16',
                           download_root=str(pathlib.Path.home() / 'Downloads' / 'whisper'))
    ecapa = EncoderClassifier.from_hparams('speechbrain/spkrec-ecapa-voxceleb', savedir=str(ROOT / 'ecapa'),
                                           run_opts={'device': 'cuda:0'}, local_strategy=LocalStrategy.COPY)

    def to16(x, rate):
        return np.interp(np.arange(0, len(x), rate / 16000), np.arange(len(x)), x).astype(np.float32)

    def embed(x16):
        with torch.no_grad():
            e = ecapa.encode_batch(torch.from_numpy(x16)[None].cuda()).squeeze().cpu().numpy()
        return e / np.linalg.norm(e)

    with open(ROOT / 'pack_check.tsv', encoding='utf-8') as f:
        clean = [r['path'] for r in csv.DictReader(f, delimiter='\t')
                 if float(r['similarity']) >= 0.6 and float(r['seconds']) <= 30]
    refs = []
    for path in clean:
        data, rate = sf.read(path, dtype='float32', always_2d=True)
        refs.append(embed(to16(data.mean(axis=1), rate)))
    ref = np.mean(refs, axis=0)
    ref /= np.linalg.norm(ref)

    print(f'{"точка":16} похожесть  ошибок  длит.  (по {len(PHRASES)} фразам)')
    for name in ['исходная', *names]:
        if name != 'исходная':
            voice.model = None
            torch.cuda.empty_cache()
            voice.model = load_model(DiT, MODEL_CFG, str(CKPTS / name),
                                     vocab_file=hf_hub_download(MODEL_REPO, 'vocab.txt'), use_ema=False)
        out = OUT / name.replace('.pt', '').replace('/', '_')
        out.mkdir(parents=True, exist_ok=True)
        sims, errs, secs = [], [], []
        for n, text in enumerate(PHRASES, 1):
            wav, rate = voice.synth(text)
            sf.write(str(out / f'{n:02}.wav'), wav, rate)
            x16 = to16(wav, rate)
            heard = ' '.join(s.text for s in whisper.transcribe(x16, language='ru', beam_size=5)[0])
            sims.append(float(embed(x16) @ ref))
            errs.append(cer(text, heard))
            secs.append(len(wav) / rate)
            if errs[-1] > 0.15:
                print(f'    {name} №{n}: «{heard.strip()}»')
        print(f'{name:16} {np.mean(sims):9.3f} {np.mean(errs) * 100:6.1f}% {np.mean(secs):5.1f} с', flush=True)


if __name__ == '__main__':
    main()
