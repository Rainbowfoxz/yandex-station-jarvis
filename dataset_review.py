"""Страница для прослушивания кандидатов в датасет голоса Джарвиса.

Берёт реплики фильмов из films\\rescored.tsv (похожесть на чистый эталон >= LOW, без повторов
записей пака) и записи пака из pack_check.tsv, которых нет в jarvis-voice. Режет звук из фильма
ffmpeg'ом в Downloads\\f5-tts\\review\\clips (24 кГц моно — частота модели) и пишет index.html.
Выбор страница сохраняет файлом jarvis_review.json в Загрузки.

    py dataset_review.py
"""
import csv
import difflib
import json
import pathlib
import re
import shutil
import subprocess

F5 = pathlib.Path.home() / 'Downloads' / 'f5-tts'
FILMS = pathlib.Path(r'D:\code\alice')
OUT = F5 / 'review'
LOW = 0.42          # ниже — почти одни чужие голоса
PAD = 0.1, 0.2      # запас до и после реплики: границы Whisper неточные, с
RATE = 24000


def norm(text):
    return re.sub(r'[^а-яё0-9 ]', '', text.lower().replace('ё', 'е')).split()


def similar(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio() >= 0.6


def main():
    with open(F5 / 'pack_check.tsv', encoding='utf-8') as f:
        pack = list(csv.DictReader(f, delimiter='\t'))
    pack_texts = [r['text'] for r in pack if float(r['similarity']) >= 0.45]
    with open(pathlib.Path(__file__).with_name('voice_catalog.tsv'), encoding='utf-8') as f:
        voice = [r['text'] for r in csv.DictReader(f, delimiter='\t') if 'jarvis-voice' in r['path']]
    items = []
    # Записи пака, которых нет в jarvis-voice (те пользователь уже отобрал сам).
    for r in pack:
        if float(r['seconds']) > 30 or any(similar(r['text'], t) for t in voice):
            continue
        items.append({'src': 'пак', 'score': float(r['similarity']), 'text': r['text'], 'path': r['path']})
    with open(F5 / 'films' / 'rescored.tsv', encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if float(r['new']) < LOW or any(similar(r['text'], t) for t in pack_texts):
                continue
            items.append({'src': r['film'], 'score': float(r['new']), 'text': r['text'],
                          'start': float(r['start']), 'end': float(r['end'])})
    items.sort(key=lambda i: -i['score'])

    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / 'clips').mkdir(parents=True)
    for n, item in enumerate(items, 1):
        name = f'{n:03}.wav'
        if 'path' in item:
            cmd = ['-i', item.pop('path')]
        else:
            a = max(item['start'] - PAD[0], 0)
            cmd = ['-ss', f'{a:.2f}', '-t', f'{item["end"] + PAD[1] - a:.2f}', '-i',
                   str(FILMS / f'{item["src"]}.mp4'), '-map', '0:a:0']
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', *cmd, '-ac', '1', '-ar', str(RATE),
                        '-c:a', 'pcm_s16le', str(OUT / 'clips' / name)], check=True)
        item.update(id=n, file=f'clips/{name}')
    page = (pathlib.Path(__file__).with_name('dataset_review.html').read_text(encoding='utf-8')
            .replace('/*ITEMS*/[]', json.dumps(items, ensure_ascii=False)))
    (OUT / 'index.html').write_text(page, encoding='utf-8')
    films = sum('start' in i for i in items)
    print(f'Реплик {len(items)} (фильмы {films}, пак {len(items) - films}): {OUT / "index.html"}')


if __name__ == '__main__':
    main()
