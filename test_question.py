"""Тест разбора: что после «Джарвиса» в тексте Vosk и что колонка сделает с ответом Gemini.

    py test_question.py
"""
from alice_say import byte_field, integer
from gemini_brain import Brain, route, split_heard
from jarvis_wake import RECOGNITION_COMPLETED, alice_event, question_after_wake

CASES = [
    ('джарвис как дела', False, 'как дела'),          # имя распознано и свободно
    ('джарвис', True, ''),                            # одно имя -> «Да, сэр»
    ('тавис сколько будет два плюс два', True, 'сколько будет два плюс два'),  # имя исказилось
    ('тавис', True, ''),
    ('какая погода завтра', False, None),            # обычная речь без имени — молчим
    ('привет джарвис', False, ''),
    ('джарвис и', True, ''),                          # огрызок вместо вопроса
    ('жарвис включи музыку', False, 'включи музыку'),
    ('Джарвис, включи мою волну.', True, 'включи мою волну'),   # регистр и знаки
    ('Джарвис?', True, ''),
    ('Джарвис, включи мою волну Джарвис, какая завтра погода?', True, 'какая завтра погода'),  # хвост прошлой
]
# Ответ Gemini на звук -> (что услышал, вид действия, текст).
ANSWERS = [
    ('ВОПРОС: какая ты нейросеть\nЯ Джарвис, сэр.', 'черновик', ('какая ты нейросеть', 'say', 'Я Джарвис, сэр.')),
    ('ВОПРОС: включи мою волну\nАЛИСА: включи мою волну', 'черновик',
     ('включи мою волну', 'alice', 'включи мою волну')),
    ('Вопрос: сколько времени\nАлиса: сколько времени.', 'черновик', ('сколько времени', 'alice', 'сколько времени')),
    ('Просто ответ без строки вопроса.', 'черновик', ('черновик', 'say', 'Просто ответ без строки вопроса.')),
    ('ВОПРОС:\n**Слушаю**, сэр.', 'черновик', ('черновик', 'say', 'Слушаю, сэр.')),   # пустой вопрос, markdown
]
bad = 0
for text, heard, want in CASES:
    got = question_after_wake(text, heard)
    ok = got == want
    bad += not ok
    print('ok ' if ok else 'FAIL', repr(text), heard, '->', repr(got), '' if ok else f'(ждали {want!r})')
for answer, fallback, want in ANSWERS:
    heard, rest = split_heard(answer, fallback)
    got = (heard, *route(rest))
    ok = got == want
    bad += not ok
    print('ok ' if ok else 'FAIL', 'Gemini:', repr(answer), '->', got, '' if ok else f'(ждали {want})')


def stream_brain(answer, size):
    """Brain без сети: модель «присылает» ответ кусками по size символов."""
    brain = object.__new__(Brain)
    brain.history, brain.last, brain.model = [], 0.0, None
    brain._stream = lambda contents: (answer[i:i + size] for i in range(0, len(answer), size))
    return brain


# Потоковый разбор даёт то же, что разбор целого ответа, как бы модель ни нарезала текст.
for answer, fallback, want in ANSWERS:
    for size in (1, 3, 7, 1000):
        kind, text, heard = stream_brain(answer, size).ask(fallback)
        got = (heard, kind, text)
        ok = got == want
        bad += not ok
        if not ok:
            print('FAIL', f'поток по {size}:', repr(answer), '->', got, f'(ждали {want})')
print('ok ', 'потоковый разбор совпал с целым для', len(ANSWERS), 'ответов и 4 нарезок')

# Первое предложение отдаётся, пока модель пишет остальное; короткие склеиваются со следующим.
LONG = 'ВОПРОС: как дела\nДа, сэр. Все системы работают штатно, сэр. Мистер Старк снова спит над кодом.'
brain = stream_brain(LONG, 4)
sent, events = [], []
brain._stream = lambda contents: (sent.append(i) or LONG[i:i + 4] for i in range(0, len(LONG), 4))
for event, value in brain.ask_stream('черновик'):
    events.append((event, value, sent[-1] + 4 < len(LONG)))   # третье — модель ещё не дописала
want = [('heard', 'как дела', True),
        ('say', 'Да, сэр. Все системы работают штатно, сэр.', True),
        ('say', 'Мистер Старк снова спит над кодом.', False)]
ok = events == want
bad += not ok
print('ok ' if ok else 'FAIL', 'по предложениям:', events, '' if ok else f'(ждали {want})')
ok = len(brain.history) == 2 and brain.history[1].parts[0].text == LONG.split('\n', 1)[1]
bad += not ok
print('ok ' if ok else 'FAIL', 'в память ушёл ответ без строки ВОПРОС')


def frame(method, text=None):
    """Кадр канала Алисы, как его видел alice_watch.py: 252 -> 18 -> {1: событие, 2 -> 15 -> 1: текст}."""
    state = byte_field(15, byte_field(1, text.encode())) if text is not None else b''
    listener = integer(1, method) + (byte_field(2, state) if text is not None else b'')
    return byte_field(252, byte_field(7, b'AliceCapability') + byte_field(18, listener)) + integer(253, 3)


EVENTS = [
    (frame(0, 'Какая сегодня'), (0, 'Какая сегодня')),
    (frame(RECOGNITION_COMPLETED), (RECOGNITION_COMPLETED, '')),
    (byte_field(252, byte_field(7, b'AliceCapability')), (0, '')),      # кадр без события слушателя
]
for data, want in EVENTS:
    got = alice_event(data)
    ok = got == want
    bad += not ok
    print('ok ' if ok else 'FAIL', 'событие Алисы ->', got, '' if ok else f'(ждали {want})')
raise SystemExit(bad)
