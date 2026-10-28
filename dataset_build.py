"""Датасет для дообучения голоса Джарвиса: реплики, отобранные на слух, и их тексты.

    ... dataset_build.py review   # собрать реплики, распознать Whisper large-v3, страница dataset\\index.html
    ... dataset_build.py export   # после проверки (Загрузки\\jarvis_dataset.json): wavs\\ + metadata.csv

Источники:
  * пак Jarvis Desktop (pack_check.tsv; это те же файлы, что jarvis-voice, но с текстом в имени файла).
    Если запись слушали на странице отбора — по отметке: 1 — как есть, 2 — очищенная Demucs, если
    после очистки отмечена 1; остальное не берём. Не слушанные — при похожести >= PACK_SIM;
  * реплики фильмов с отметкой 1 на первой странице или после очистки.
Тексты: пак — имя файла (написано человеком), фильм — правка пользователя, иначе Whisper large-v3.
Числа переводятся в слова (именительный падеж — падеж проверяет человек, реплика помечена).
Запускать питоном из окружения F5-TTS.
"""
import csv
import json
import os
import pathlib
import re
import shutil
import sys

F5 = pathlib.Path.home() / 'Downloads' / 'f5-tts'
os.environ.setdefault('HF_HOME', str(F5 / 'hf'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

HERE = pathlib.Path(__file__).parent
DL = pathlib.Path.home() / 'Downloads'
OUT = F5 / 'dataset'
RATE = 24000
PACK_SIM = 0.45
EDGE = 0.1          # тишины по краям оставляем, с
TARGET_RMS = 0.08   # одна громкость для всех реплик (~-22 dBFS)
SKIP_PACK = ('Рассказ про Ивана Ванко',   # тот же звук, что реплики фильма №3, 5, 9, 11, 20 — они разбиты по фразам
             'Джарвис - приветствие (Песня целиком)')
LABELS = {1: 'Текст верный', 2: 'Сомневаюсь', 3: 'Убрать реплику'}
# Ручная доводка после проверки (№ на странице текстов, 28.10): громкость по 0,1 с + метки слов Whisper.
CUTS = {26: (0, 3.2), 33: (0, 2.1), 61: (2.1, None), 65: (0.75, None), 86: (0, 1.15)}   # хвосты и чужой звук, с
DROP = {85: 'начинается с середины фразы', 87: 'обрезана на полуслове'}


def norm(text):
    return re.sub(r'[^а-яё0-9 ]', '', text.lower().replace('ё', 'е')).split()


def similar(a, b):
    import difflib
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio() >= 0.6


def page_items(path):
    html = path.read_text(encoding='utf-8')
    return json.loads(re.search(r'const ITEMS = (\[.*?\]);\n', html, re.S).group(1))


def pack_text(name):
    """Имя файла пака -> текст: без «(второй)», « 2», смайлов."""
    text = re.sub(r'\s*\((второй|второй вариант)\)|\s*=\)|\s+2$', '', name).strip()
    return text[0].upper() + text[1:] + ('' if text[-1] in '.!?' else '.')


def spell_numbers(text):
    """«24%» -> «двадцать четыре процента» (падеж числа — на проверку человеку)."""
    from num2words import num2words

    def pct(m):
        n = int(m.group(1))
        word = 'процент' if n % 10 == 1 and n % 100 != 11 else (
            'процента' if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else 'процентов')
        return f'{num2words(n, lang="ru")} {word}'
    def ordinal(m):   # «в 1963 году», «в 67-м» — порядковое в предложном падеже
        return num2words(int(m.group(1)), lang='ru', to='ordinal', case='prepositional')
    text = re.sub(r'(\d+)\s*%', pct, text)
    text = re.sub(r'(?<=\bв )(\d+)(?=\s+году\b)', ordinal, text)
    text = re.sub(r'(\d+)-м\b', ordinal, text)
    return re.sub(r'\d+', lambda m: num2words(int(m.group()), lang='ru'), text)


def load(path):
    data, rate = sf.read(str(path), dtype='float32', always_2d=True)
    mono = data.mean(axis=1)
    if rate != RATE:
        mono = np.interp(np.arange(0, len(mono), rate / RATE), np.arange(len(mono)), mono).astype(np.float32)
    return mono


def collect():
    r1 = json.loads((DL / 'jarvis_review.json').read_text(encoding='utf-8'))
    r2 = {i['review_id']: i for i in json.loads((DL / 'jarvis_review2.json').read_text(encoding='utf-8'))}
    shown1 = {i['id']: i['text'] for i in page_items(F5 / 'review' / 'index.html')}
    items = []

    def verdict_audio(item):
        """Звук реплики по отметкам: (путь, текст правки или None) или None — не берём."""
        if item['verdict'] == 1:
            return F5 / 'review' / item['file'], item['text'] if item['text'] != shown1[item['id']] else None
        if item['verdict'] == 2 and item['id'] in r2 and r2[item['id']]['verdict'] == 1:
            cleaned = r2[item['id']]
            return F5 / 'review2' / cleaned['file'], (cleaned['text'] if cleaned['text'] != item['text'] else None)
        return None

    reviewed_pack = {i['text']: i for i in r1 if i['src'] == 'пак'}
    with open(F5 / 'pack_check.tsv', encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['text'] in SKIP_PACK:
                continue
            if r['text'] in reviewed_pack:
                got = verdict_audio(reviewed_pack[r['text']])
                if got is None:
                    continue
                path, edit = got
            elif float(r['similarity']) >= PACK_SIM:
                path, edit = pathlib.Path(r['path']), None
            else:
                continue
            items.append({'src': 'пак', 'audio': str(path), 'text': edit or pack_text(r['text']), 'human': True})
    for item in r1:
        if item['src'] == 'пак':
            continue
        got = verdict_audio(item)
        if got is None:
            continue
        path, edit = got
        if any(similar(item['text'], i['text']) for i in items):
            continue
        items.append({'src': item['src'], 'start': item.get('start'), 'audio': str(path),
                      'text': edit or item['text'], 'human': edit is not None})
    return items


def review():
    import torch  # noqa: F401  — DLL CUDA для faster-whisper
    from faster_whisper import WhisperModel
    items = collect()
    whisper = WhisperModel('large-v3', device='cuda', compute_type='float16', download_root=str(DL / 'whisper'))
    (OUT / 'clips').mkdir(parents=True, exist_ok=True)
    for n, item in enumerate(items, 1):
        wav = load(item['audio'])
        name = f'{n:03}.wav'
        sf.write(str(OUT / 'clips' / name), wav, RATE, subtype='PCM_16')
        x16 = np.interp(np.arange(0, len(wav), RATE / 16000), np.arange(len(wav)), wav).astype(np.float32)
        segments, _ = whisper.transcribe(x16, language='ru', beam_size=5)
        asr = ' '.join(s.text.strip() for s in segments).strip()
        notes = []
        if not item['human']:
            item['small'] = item['text']
            item['text'] = asr or item['text']
            if not similar(asr, item['small']):
                notes.append(f'другой Whisper слышал: «{item["small"]}»')
        elif asr and not similar(asr, item['text']):
            notes.append(f'Whisper слышит: «{asr}»')
        twin = next((i['id'] for i in items[:n - 1] if similar(i['text'], item['text'])
                     or similar(i['asr'], asr)), None)
        if twin:
            notes.append(f'повтор №{twin} — оставь лучшую запись')
        if re.search(r'\d', item['text']):
            item['digits'] = item['text']
            item['text'] = spell_numbers(item['text'])
            notes.append(f'было «{item["digits"]}» — проверь падеж числа')
        item.update(id=n, file=f'clips/{name}', asr=asr, score=len(wav) / RATE, note='; '.join(notes))
        print(f'{n}/{len(items)} {item["text"][:70]}', flush=True)
    page = ((HERE / 'dataset_review.html').read_text(encoding='utf-8')
            .replace('/*ITEMS*/[]', json.dumps(items, ensure_ascii=False))
            .replace("/*SAVE_AS*/'jarvis_review.json'", "'jarvis_dataset.json'")
            .replace("/*LABELS*/{ 1: 'Джарвис, чисто', 2: 'Джарвис, шум', 3: 'Не он' }",
                     json.dumps(LABELS, ensure_ascii=False))
            .replace('<h1>Отбор реплик Джарвиса</h1>', '<h1>Тексты датасета</h1>')
            .replace('похожесть ${item.score.toFixed(2)}', '${item.score.toFixed(1)} с'))
    (OUT / 'index.html').write_text(page, encoding='utf-8')
    total = sum(i['score'] for i in items)
    print(f'Реплик {len(items)}, {total / 60:.1f} мин: {OUT / "index.html"}')


def trim(wav):
    """Обрезать тишину по краям (окна 10 мс, -40 дБ от пика), оставить EDGE; выровнять громкость."""
    win = RATE // 100
    n = len(wav) // win
    rms = np.sqrt(np.mean(wav[:n * win].reshape(n, win) ** 2, axis=1)) + 1e-9
    loud = np.flatnonzero(20 * np.log10(rms / rms.max()) > -40)
    edge = int(EDGE * 100)
    wav = wav[max(loud[0] - edge, 0) * win:min(loud[-1] + edge + 1, n) * win]
    gain = TARGET_RMS / (np.sqrt(np.mean(wav ** 2)) + 1e-9)
    return np.clip(wav * min(gain, 0.99 / (np.abs(wav).max() + 1e-9)), -1, 1)


def export():
    sys.path.insert(0, str(HERE))
    from jarvis_voice import MODEL_REPO, RUAccent, ROOT, hf_hub_download
    items = json.loads((DL / 'jarvis_dataset.json').read_text(encoding='utf-8'))
    keep = [i for i in items if i['verdict'] == 1 and i['text'].strip() and i['id'] not in DROP]
    for item in keep:   # человек мог вернуть цифры при правке
        item['text'] = spell_numbers(re.sub(r'\s+', ' ', item['text']).strip())
    vocab = set(open(hf_hub_download(MODEL_REPO, 'vocab.txt'), encoding='utf-8').read().split('\n'))
    latin = [i['id'] for i in keep if re.search(r'[A-Za-z]', i['text'])]
    unknown = {c for i in keep for c in i['text'] if c not in vocab}
    if latin or unknown:
        sys.exit(f'Латиница в №{latin}, нет в словаре модели: {sorted(unknown)} — исправить тексты')
    accent = RUAccent()
    accent.load(omograph_model_size='turbo3.1', use_dictionary=False, tiny_mode=False, workdir=str(ROOT / 'ruaccent'))
    accent.accent_model.tokenizer.model_input_names = ['input_ids', 'token_type_ids', 'attention_mask']
    shutil.rmtree(OUT / 'wavs', ignore_errors=True)   # без хвостов прошлых выгрузок
    (OUT / 'wavs').mkdir(parents=True)
    total = 0.0
    with open(OUT / 'metadata.csv', 'w', encoding='utf-8', newline='') as f:
        f.write('audio_file|text\n')
        for item in keep:
            wav = load(OUT / item['file'])
            if item['id'] in CUTS:
                a, b = CUTS[item['id']]
                wav = wav[int(a * RATE):int(b * RATE) if b else None]
            wav = trim(wav)
            path = OUT / 'wavs' / pathlib.Path(item['file']).name
            sf.write(str(path), wav, RATE, subtype='PCM_16')
            f.write(f'{path.as_posix()}|{accent.process_all(item["text"])}\n')
            total += len(wav) / RATE
    print(f'В датасете {len(keep)} реплик из {len(items)}, {total / 60:.1f} мин: {OUT / "metadata.csv"}')


if __name__ == '__main__':
    {'review': review, 'export': export}[sys.argv[1]]()
