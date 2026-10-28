"""Сквозной тест слушателя без колонки: человек говорит «Джарвис, …» поверх музыки.

Поток: music6.raw (музыка, её опорные каналы) + микрофон из phrases6.raw («Джарвис, включи мою
волну», «Джарвис, какая завтра погода») без опоры — так звучит человек в комнате. Звук идёт в темпе
реального времени, как с колонки. Под музыкой на «Джарвис» Алиса должна сразу начать слушать
(capture), её распознанный текст — уйти в Gemini.

    py test_pipeline.py              # заглушка Gemini
    py test_pipeline.py --gemini     # настоящий Gemini
    py test_pipeline.py --raw        # без вычитания эха — «Джарвис» под музыкой не ловится
"""
import io
import sys
import threading
import time

import numpy as np
import vosk

import jarvis_wake as jw

GAIN = 0.5   # голос человека тише, чем колонка говорила себе в микрофон
# Что распознала бы Алиса после «Джарвис» — по порядку фраз в phrases6.raw.
ALICE_HEARS = ['включи мою волну', 'какая завтра погода']

music = np.frombuffer(open('music6.raw', 'rb').read(), '<i2').reshape(-1, jw.CHANNELS).astype(np.float64)
voice = np.frombuffer(open('phrases6.raw', 'rb').read(), '<i2').reshape(-1, jw.CHANNELS)[:, jw.MIC]
# 13 с: третья фраза («Джарвис, будильник…») начинается на 14-й секунде — в запись не попадает.
n = min(len(music), len(voice) - 48000 * 2, 48000 * 13)
frames = music[:n].copy()
frames[:, jw.MIC] += voice[48000 * 2:48000 * 2 + n] * GAIN
stream = np.clip(frames, -32768, 32767).astype('<i2').tobytes()


class Paced(io.BytesIO):
    """Отдаёт поток в темпе реального времени (48 кГц × 12 байт), как колонка."""

    def read(self, size=-1):
        block = super().read(size)
        time.sleep(len(block) / (48000 * jw.FRAME))
        return block


class FakeSession:
    def __init__(self):
        self.audio = type('P', (), {'stdout': Paced(stream)})()
        self.asked = []
        self.busy = False
        self.capturing = False
        self.capture_deadline = 0.0
        self.on_question = None
        self.captures = 0

    def capture(self):
        # Как настоящая Алиса: слушает ~1,5 с, распознаёт и отдаёт текст через on_question.
        self.capturing, self.capture_deadline = True, time.monotonic() + jw.CAPTURE_MAX
        text = ALICE_HEARS[min(self.captures, len(ALICE_HEARS) - 1)]
        self.captures += 1
        self.asked.append('<Алиса слушает>')

        def recognized():
            time.sleep(1.5)
            self.capturing = False
            self.on_question(text)
        threading.Thread(target=recognized, daemon=True).start()

    def ask(self, text):
        self.asked.append(text)

    def say(self, text):
        self.asked.append(f'<сказано: {text}>')

    def speech(self, rate):
        session = self

        class Speech:   # голос Джарвиса: запоминаем, сколько звука ушло бы на колонку
            failed = False
            started = time.monotonic()
            seconds = 0.0

            def add(self, pcm):
                self.seconds += len(pcm) / 2 / rate
                session.asked.append(f'<Джарвис: {len(pcm) / 2 / rate:.1f} с>')

            def close(self):
                pass
        return Speech()

    def think(self, on):
        pass

    def reply(self, name):
        self.asked.append('<Да, сэр>')
        return 0

    def listen(self):
        pass


class FakeBrain:
    """Как Gemini: команды колонке — Алисе, остальное — свой ответ."""
    questions = []

    def ask_stream(self, question, audio=None):
        self.questions.append(question)
        yield 'heard', question
        if 'волн' in question:
            yield 'alice', question
        else:
            yield 'say', f'ответ Джарвиса на «{question}».'


class LiveBrain(jw.Brain):
    """Настоящий Gemini; запоминает, что он получил."""
    questions = []

    def ask_stream(self, question, audio=None):
        heard = question
        for event, value in super().ask_stream(question, audio):
            heard = value if event == 'heard' else heard
            yield event, value
        self.questions.append(heard)


if '--raw' in sys.argv:
    jw.MicDecoder.__init__.__defaults__ = (False,)
vosk.SetLogLevel(-1)
model = vosk.Model(str(jw.MODEL))
session = FakeSession()
brain = LiveBrain() if '--gemini' in sys.argv else FakeBrain()
jw.run(session, vosk.KaldiRecognizer(model, 16000, jw.GRAMMAR), vosk.KaldiRecognizer(model, 16000), brain)
for thread in threading.enumerate():   # распознавание «Алисы» и ответы Gemini идут в своих потоках
    if thread is not threading.current_thread():
        thread.join(20)
print('Алиса слушала раз:', session.captures)
print('Gemini получил:', type(brain).questions)
print('Колонке ушло:', session.asked)
if '--raw' in sys.argv:
    sys.exit(0 if session.captures == 0 else 1)
ok = session.captures == 2 and len(type(brain).questions) == 2
if '--gemini' in sys.argv:
    ok = ok and 'включи мою волну' in session.asked
sys.exit(0 if ok else 1)
