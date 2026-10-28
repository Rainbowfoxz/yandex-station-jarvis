"""Gemini в роли Джарвиса: отвечает на вопросы, а команды колонке переправляет Алисе.

Ответ модели — либо обычный текст (его колонка читает голосом Алисы), либо строка
«АЛИСА: <команда>» — тогда команда уходит Алисе текстом (музыка, громкость, таймеры, умный дом).
Вопрос можно отдать звуком (audio=): Gemini сам разбирает речь — точнее Vosk и Whisper на живом
голосе — и первой строкой пишет «ВОПРОС: …», что услышал. Звук — после вычитания музыки (aec.py):
на сыром микрофоне под музыкой он слышит «Не берешь трубку» вместо «включи мою волну».
Ключ — GEMINI_API_KEY из окружения или из .env проекта LMS.

    py gemini_brain.py "кто написал войну и мир"     # проверка без колонки
"""
import base64
import io
import json
import os
import pathlib
import queue
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import wave

from jarvis_settings import GEMINI_ENV_FILE as ENV_FILE  # noqa: E402
# REST напрямую, без google-genai: библиотека с зависимостями занимала большую часть памяти слушателя
# и полминуты его запуска, а от неё нужен один запрос. Ответ — потоком (SSE), как generate_content_stream.
API = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse'
# Lite — первой: для колонки скорость важнее (1–6 с), и она отвечала, когда старшие давали 503/тайм-аут.
# Слабее в фактах — при её сбое пробуем старшие. У каждой версии своя очередь и квота у Google: 28.10
# lite (504) и flash-latest (503) лежали, а 3.6 и 3.5 отвечали на голосовой вопрос за 2,6–3,2 с.
# gemini-2.5-flash(-lite) новым ключам недоступны (404); gemini-3-flash-preview повторяет «ВОПРОС:» в ответе.
MODELS = ('gemini-flash-lite-latest', 'gemini-3.6-flash', 'gemini-3.5-flash', 'gemini-3.1-flash-lite',
          'gemini-flash-latest', 'gemini-3.8-flash')
# Уровень размышления: MINIMAL принимают lite, 3.6 и 3.5 (проверено 28.10); flash-latest и 3.8 отвечают
# 400 «not supported» — до 28.10 запасные модели из-за этого не срабатывали ни разу. Остальным — LOW.
MINIMAL_THINKING = {'gemini-flash-lite-latest', 'gemini-3.6-flash', 'gemini-3.5-flash'}
TIMEOUT_MS = 10000            # меньше сервер не принимает (400 «minimum deadline is 10s»)
# Тайм-аут модели — 10 с ожидания; две такие подряд колонка молчала бы 20+ с. Новые модели после этого
# срока не пробуем — вопрос уходит Алисе.
TRY_SECONDS = 15
RACE = 3                      # сколько моделей спрашиваем одновременно (у каждой своя квота 5 в минуту)
# Удачная запасная модель пробуется первой, но через 10 мин снова начинаем с основной: lite быстрее (1–3 с
# против 3–5 с у 3.6), а её сбои обычно проходят. Пока она лежит — раз в 10 мин один ответ на 10 с дольше.
BACK_TO_PRIMARY = 600
# Поиск Google на этом ключе даёт 429 (квота) на всех моделях; включить — JARVIS_SEARCH=1.
SEARCH = bool(os.environ.get('JARVIS_SEARCH'))
MEMORY_SECONDS = 300          # разговор помнится, пока между вопросами меньше 5 минут
MAX_TURNS = 10                # и не больше 10 последних пар вопрос-ответ
ALICE = 'АЛИСА:'
HEARD = 'ВОПРОС:'
AUDIO_PROMPT = f"""Это голосовое сообщение пользователя. В начале — обращение «Джарвис» (может звучать
искажённо). Если в записи несколько фраз, важна последняя. Первой строкой напиши «{HEARD} <что сказано
после обращения>», со следующей строки — ответ по обычным правилам."""

PERSONA = f"""Ты — Джарвис, дворецкий-ИИ из «Железного человека», живёшь в умной колонке.
Говоришь по-русски, вежливо и сдержанно, с лёгкой британской иронией, обращаешься «сэр».
Шутки и отсылки — только когда это разрешает подсказка в скобках после вопроса, не больше одной
сухой фразы и не вместо сути ответа. Твой основной юмор — сухая ирония над привычками и
экспериментами самого сэра одной фразой в конце ответа, например: «Физические законы здесь
соблюдаются, по крайней мере, пока вы не пытаетесь снова улучшить тостер» — это образец тона,
каждый раз придумывай новую, про тостер не повторяй. Отсылки к вайбкодингу (код, написанный нейросетью «на ощущениях»,
«оно как-то работает, не трогайте», промпт вместо ТЗ, коммит в пятницу вечером, «просто попроси ИИ
переписать всё заново») уместны только в разговоре о программировании, компьютерах, ИИ и работе
над проектами — про природу, еду, историю, науку и быт их не бывает. Мистера Старка упоминаешь,
только если подсказка это разрешает. В командах колонке и серьёзных вопросах (здоровье, безопасность)
не шутишь никогда.
Твой ответ читается вслух синтезатором речи, поэтому:
- 1–3 коротких предложения, без списков, markdown, эмодзи и ссылок;
- числа, даты и единицы пиши так, как их произносят.
Колонкой управляет Алиса. Если пользователь просит действие с колонкой или домом — музыку
(включить, остановить, следующий трек, «мою волну»), громкость, таймер, будильник, напоминание,
погоду, время, свет и умный дом, — ответь ровно одной строкой «{ALICE} <команда для Алисы>»,
например «{ALICE} включи мою волну» или «{ALICE} поставь таймер на 10 минут». Больше ничего не пиши.
На всё остальное отвечай сам."""
SEARCH_HINT = '\nДля свежих фактов и новостей пользуйся поиском.'
# Модель не считает свои ответы: «в каждом третьем» в характере давало шутку в 4 ответах из 6.
# Поэтому шутку (не больше одной) выбирает жребий в коде и подсказывает к каждому вопросу.
# «Если к месту» модель не соблюдает: с разрешением вайбкодинг был в 5 ответах из 6 про планеты,
# гречку и Толстого (28.10) — тема названа явно. Старк без жребия был в 2–4 ответах из 8 — слишком часто.
# Пользователю нравится ирония в духе «пока вы не пытаетесь снова улучшить тостер» — она основная.
NO_STARK = ' Мистера Старка не упоминай.'
IRONY = '(Можно закончить ответ одной сухой ироничной фразой о привычках сэра, без вайбкодинга.' + NO_STARK + ')'
JOKES = [   # (доля ответов, подсказка); в остальных — NO_JOKE
    (0.3, IRONY),
    (0.2, '(Если вопрос о программировании, компьютерах, ИИ или работе над проектом — можно одну шутку '
          'про вайбкодинг. Если о чём-то другом — вместо неё можно одну сухую ироничную фразу о привычках '
          'сэра, без кода и нейросетей.' + NO_STARK + ')'),
    (0.1, '(Можно одну сухую отсылку к мистеру Старку, без вайбкодинга.)'),
]
NO_JOKE = '(В этом ответе без шуток: ни иронии, ни вайбкодинга, ни мистера Старка.)'


def joke_hint():
    """Жребий: какая шутка (одна или никакой) разрешена в этом ответе."""
    draw = random.random()
    for share, hint in JOKES:
        if draw < share:
            return hint
        draw -= share
    return NO_JOKE
SENTENCE_END = re.compile(r'(?<=[.!?…])\s+')
# Короткие предложения («Да, сэр.») склеиваем со следующим: синтез стоит почти одинаково,
# а на совсем коротком тексте голос звучит хуже.
MIN_PIECE = 30


def api_key():
    key = os.environ.get('GEMINI_API_KEY')
    if not key and ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding='utf-8').splitlines():
            name, _, value = line.partition('=')
            if name.strip() == 'GEMINI_API_KEY':
                key = value.strip().strip('"\'')
    if not key:
        raise RuntimeError(f'Нет GEMINI_API_KEY ни в окружении, ни в {ENV_FILE}')
    return key


class Stream:
    """Ответ модели кусками текста (SSE «data: {…}»); close() обрывает соединение — Google перестаёт писать."""

    def __init__(self, response):
        self.response = response

    def __iter__(self):
        for raw in self.response:
            line = raw.decode('utf-8', errors='replace').strip()
            if not line.startswith('data:'):
                continue
            chunk = json.loads(line[5:])
            for candidate in chunk.get('candidates', [])[:1]:
                for part in candidate.get('content', {}).get('parts', []):
                    if part.get('text') and not part.get('thought'):
                        yield part['text']

    def close(self):
        try:
            self.response.close()
        except Exception:   # закрываем из другого потока, пока тот читает, — ошибки не важны
            pass


class Brain:
    def __init__(self):
        # Ключ «AQ.…» — новый формат ключей Gemini API (не Vertex: там API в проекте выключен, 403).
        # Автоповторов нет: при сбое сразу пробуем другую модель, а не ждём паузы 1–8 с.
        self.key = api_key()
        self.model = None
        self.fallback_at = 0.0    # когда ответила запасная модель (а не MODELS[0])
        self.history = []     # {'role': 'user'|'model', 'parts': [...]} по очереди
        self.last = 0.0

    def _open(self, model, contents):
        """Запрос к модели -> Stream. Ошибка HTTP -> RuntimeError «код СТАТУС. сообщение» (как писал SDK)."""
        # Для колонки скорость важнее рассуждений: с ними ответ шёл 4–5 с вместо долей секунды.
        body = {'system_instruction': {'parts': [{'text': PERSONA + (SEARCH_HINT if SEARCH else '')}]},
                'contents': contents,
                'generationConfig': {'temperature': 0.7, 'maxOutputTokens': 400, 'thinkingConfig': {
                    'thinkingLevel': 'MINIMAL' if model in MINIMAL_THINKING else 'LOW'}}}
        if SEARCH:
            body['tools'] = [{'google_search': {}}]
        request = urllib.request.Request(API.format(model=model), data=json.dumps(body).encode('utf-8'),
                                         headers={'x-goog-api-key': self.key,
                                                  'Content-Type': 'application/json'})
        try:
            return Stream(urllib.request.urlopen(request, timeout=TIMEOUT_MS / 1000))
        except urllib.error.HTTPError as error:
            try:
                info = json.loads(error.read()).get('error', {})
            except ValueError:
                info = {}
            raise RuntimeError(f'{error.code} {info.get("status", error.reason)}. '
                               f'{info.get("message", "")[:120]}') from None

    def ask(self, question, audio=None):
        """Вопрос текстом или звуком (PCM S16_LE 16 кГц моно) -> (вид, текст, что услышано):
        ('say', ответ, вопрос) или ('alice', команда, вопрос)."""
        kind, texts, heard = 'say', [], question
        for event, value in self.ask_stream(question, audio):
            if event == 'heard':
                heard = value
            else:
                kind = event
                texts.append(value)
        return kind, ' '.join(texts), heard

    def ask_stream(self, question, audio=None):
        """То же, что ask, но по мере ответа модели: ('heard', что услышано) — если модель написала
        «ВОПРОС: …»; затем ('alice', команда) или ('say', кусок) по законченным предложениям —
        первый кусок можно озвучивать, пока модель пишет остальное."""
        if time.monotonic() - self.last > MEMORY_SECONDS:
            self.history.clear()
        joke = joke_hint()
        if audio is None:
            parts = [{'text': f'{question}\n{joke}'}]
        else:
            parts = [{'inline_data': {'mime_type': 'audio/wav', 'data': base64.b64encode(to_wav(audio)).decode()}},
                     {'text': f'{AUDIO_PROMPT}\n{joke}'}]
        full, buf, kind, heard = '', '', None, question
        for chunk in self._stream(self.history + [{'role': 'user', 'parts': parts}]):
            full += chunk
            buf += chunk
            while kind is None:              # начало ответа: «ВОПРОС:», «АЛИСА:» или сразу речь
                line, newline, rest = buf.partition('\n')
                head = line.strip().upper()
                if not newline and (prefix_of(head, HEARD) or prefix_of(head, ALICE)):
                    break                    # строка ещё не дописана — ждём
                if head.startswith(HEARD):
                    heard = line.strip()[len(HEARD):].strip() or question
                    buf = rest
                    yield 'heard', heard
                elif head.startswith(ALICE) and line.strip()[len(ALICE):].strip(' .'):
                    kind = 'alice'
                    yield 'alice', line.strip()[len(ALICE):].strip(' .')
                elif not head and newline:
                    buf = rest
                else:
                    kind = 'say'
            if kind == 'say':
                *ready, buf = SENTENCE_END.split(buf)
                piece = ''
                for sentence in ready:
                    piece = f'{piece} {sentence}'.strip()
                    if len(piece) >= MIN_PIECE:
                        yield 'say', clean(piece)
                        piece = ''
                buf = f'{piece} {buf}'.strip() if piece else buf
        if kind is None:                     # ответ кончился, не дописав «ВОПРОС:»/«АЛИСА:» до строки
            kind, text = route(buf)
            if text:
                yield kind, text
        elif kind == 'say' and buf.strip():
            yield 'say', clean(buf)
        # В память — текстом: звук прошлых вопросов модели не нужен, а запрос с ним тяжелее.
        answer = split_heard(full.strip(), question)[1]
        self.history = (self.history + [{'role': 'user', 'parts': [{'text': heard}]},
                                        {'role': 'model', 'parts': [{'text': answer}]}])[-2 * MAX_TURNS:]
        self.last = time.monotonic()

    def _stream(self, contents):
        # Удачная модель — первой, но спрашиваем сразу RACE моделей и берём ту, что первой начала отвечать:
        # по очереди зависшая модель съедала 10 с тайм-аута (28.10: lite молчала, ответ через 14–16 с).
        # Сбой модели (503, 429, тайм-аут) — вместо неё в гонку встаёт следующая по списку.
        # Сбой обычно приходит до первого куска; посреди ответа — ошибка.
        if self.model != MODELS[0] and time.monotonic() - self.fallback_at > BACK_TO_PRIMARY:
            self.model = None
        waiting = sorted(MODELS, key=lambda m: m != self.model)
        results = queue.Queue()
        opened, won = [], threading.Event()   # проигравшие соединения обрываем: модель перестаёт писать впустую

        def start(model):
            try:
                stream = self._open(model, contents)
                opened.append(stream)
                if won.is_set():
                    stream.close()
                    return
                chunks = iter(stream)
                results.put((model, stream, chunks, next(chunks, None), None))
            except Exception as exc:   # модели нет на этом ключе, перегрузка, тайм-аут, оборвали сами
                results.put((model, None, None, None, exc))

        running, error, give_up = 0, None, time.monotonic() + TRY_SECONDS
        while running < RACE and waiting:
            threading.Thread(target=start, args=(waiting.pop(0),), daemon=True).start()
            running += 1
        while running:
            try:
                model, stream, chunks, first, exc = results.get(timeout=max(0.1, give_up - time.monotonic()))
            except queue.Empty:
                break
            running -= 1
            if exc is not None:
                error = exc
                if waiting and time.monotonic() < give_up:
                    threading.Thread(target=start, args=(waiting.pop(0),), daemon=True).start()
                    running += 1
                continue
            won.set()
            for other in list(opened):
                if other is not stream:
                    other.close()
            if model != MODELS[0] and model != self.model:   # отсчёт 10 мин — с перехода на запасную
                self.fallback_at = time.monotonic()
            self.model = model
            try:
                if first is not None:
                    yield first
                    yield from chunks
            finally:
                stream.close()
            return
        won.set()
        for other in list(opened):
            other.close()
        raise error or TimeoutError(f'ни одна модель не ответила за {TRY_SECONDS} с')


def prefix_of(head, marker):
    """Начало строки ещё может оказаться маркером («ВОП» -> «ВОПРОС:») или уже им начинается."""
    return marker.startswith(head) or head.startswith(marker)


def clean(text):
    return ' '.join(text.replace('*', '').split())


def to_wav(pcm):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return buf.getvalue()


def split_heard(answer, fallback):
    """«ВОПРОС: …» первой строкой -> (услышанный вопрос, остальной ответ)."""
    lines = answer.splitlines()
    for i, line in enumerate(lines):
        if line.strip().upper().startswith(HEARD):
            heard = line.strip()[len(HEARD):].strip()
            return heard or fallback, '\n'.join(lines[:i] + lines[i + 1:]).strip()
    return fallback, answer


def route(answer):
    """Ответ модели -> ('alice', команда) или ('say', текст для озвучки)."""
    for line in answer.splitlines():
        line = line.strip()
        if line.upper().startswith(ALICE):
            command = line[len(ALICE):].strip(' .')
            if command:
                return 'alice', command
    text = ' '.join(answer.replace('*', '').split())
    return 'say', text


if __name__ == '__main__':
    brain = Brain()
    for q in sys.argv[1:] or ['кто ты?']:
        t = time.perf_counter()
        print(q, '->', brain.ask(q)[:2], f'({time.perf_counter() - t:.1f} с, {brain.model})')
