"""«Джарвис» без «Алисы»: слушаем микрофон колонки.

    py jarvis_wake.py            # работает, пока не нажать Ctrl+C

«Джарвис, как дела» одной фразой — звук фразы уходит Gemini (gemini_brain.py, характер Джарвиса),
он сам разбирает речь; ответ колонка читает голосом Джарвиса (jarvis_tts_server.py, F5-TTS; не запущен —
голосом Алисы), команды колонке (музыка, громкость, таймеры) Gemini возвращает Алисе. Короткие частые команды («стоп», «тише») идут Алисе сразу.
Одно «Джарвис» — ответ «Да, сэр», вопрос слушает и распознаёт Алиса, отвечает Gemini.
Под музыкой — сразу на «Джарвис» Алиса приглушает музыку и слушает (как на своё имя), её распознанный
текст перехватывается, её ответ отменяется, а вопрос уходит Gemini. Говорить: «Джарвис» — сигнал — вопрос.

Звук: ALSA-плагин file копирует вход Алисы в /tmp/jarvis на колонке (см. mic/make_asound.py),
/data/opt/jarvis/pull.sh отдаёт его по SSH. Из микрофона вычитается то, что играет динамик
(aec.py, по опорным каналам), — иначе под музыкой имя не слышно.
Офлайн только Vosk: грамматика из одного слова решает, был ли «Джарвис», свободный Vosk отмечает
конец фразы и ловит короткие команды. Текст вопроса Vosk и Whisper путали на живом голосе
(«какая ты нироситая»), поэтому вопрос Gemini получает звуком.
"""
import io
import collections
import json
import os
import pathlib
import queue
import random
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import wave

import numpy as np

import alice_say
from aec import REF_ACTIVE, EchoCanceller
# vosk и scipy грузятся только в обычном режиме (main, MicDecoder): в --station имя ловит колонка,
# а эти библиотеки занимали ~200 МБ памяти зря.
from ask_text import text_frame
from gemini_brain import Brain
from jarvis import SSH
from jarvis_listen import CANCEL_DIALOG_AND_CLEAR_QUEUE, DURATIONS, GST, STOP_TTS, WAKE, alice_method
from say import LOCAL_PORT, open_tunnel

MODEL = pathlib.Path.home() / 'Downloads' / 'vosk' / 'vosk-model-small-ru-0.22'
GRAMMAR = '["джарвис", "[unk]"]'
CHANNELS, MIC = 6, 0          # Алиса открывает вход с 6 каналами: 4 микрофона + 2 опорных динамика
REFS = (4, 5)                 # опорные каналы: что сейчас играет динамик
FRAME = CHANNELS * 2
CHUNK = 172032                # кусок приёмника (кратен кадру в 12 байт)
COOLDOWN = 3.0                # после срабатывания: своя же реплика и эхо не должны будить снова
UTTER_BYTES = 16000 * 2 * 8   # Gemini получает не больше 8 с последней фразы
# Частые короткие команды: если Vosk услышал ровно их, Алиса получает их сразу, без Gemini.
QUICK = {'стоп', 'пауза', 'дальше', 'назад', 'громче', 'тише', 'продолжи', 'продолжай', 'хватит',
         'выключи', 'следующий трек', 'следующая песня', 'включи музыку', 'включи мою волну'}
MUSIC_HOLD = 1.5              # «играет музыка», если динамик звучал за последние 1,5 с
CAPTURE_MAX = 12.0            # дольше распознавания Алисы не ждём
RECOGNITION_COMPLETED = 7     # remoting.proto, AliceCapabilityListenerMethod
THINK_MS = 15000             # анимация «думаю» пока ждём Gemini (гасится, как только ответ готов)
RECONNECT = 3.0              # пауза перед переподключением, если поток с колонки оборвался
START_CONVERSATION = 2        # remoting.proto, AliceCapabilityMethod
# Голос Джарвиса: jarvis_tts_server.py (F5-TTS). Не запущен или ошибся — ответ читает Алиса.
TTS_URL = 'http://127.0.0.1:19890/tts'
TTS_TIMEOUT = 15.0
# Сырой PCM из stdin (fdsrc): rawaudioparse на колонке нет, формат задаёт фильтр caps.
STREAM_GST = GST.replace('filesrc location=/data/opt/jarvis/og/{name}.wav ! decodebin',
                         'fdsrc fd=0 ! audio/x-raw,format=S16LE,rate={rate},channels=1,layout=interleaved')
SPEAK_MAX_MS = 60000              # дольше ответ не звучит: подсветка и ожидание проигрыша
# maind объявляет конец потока, когда последний кусок ушёл в буфер ALSA, и сразу закрывается — последние
# ~0,3 с не звучат (замер на колонке 28.10: «улучшить то…» без «стер»). Синтез обрезан до 0,05 с тишины,
# поэтому в хвост дописываем тишину: теряется она, а не последний слог.
TAIL_SILENCE = 0.8
LAST_QUESTION = pathlib.Path(os.environ.get('TEMP', '.')) / 'jarvis_question.wav'   # звук последнего вопроса
ASK_FILE = pathlib.Path(os.environ.get('TEMP', '.')) / 'jarvis_ask.txt'   # вопрос текстом для проверок
# («!текст» — сразу Алисе, мимо Gemini: проверка той же дороги, что при сбое Gemini)
ALICE_IDLE = 60.0            # соединение с Алисой простояло дольше — перед отправкой открываем новое
DEBUG = bool(os.environ.get('JARVIS_DEBUG'))  # печатать поток и всё распознанное



class MicDecoder:
    """Поток 6-канальных кадров 48 кГц кусками любой длины -> PCM 16 кГц одного микрофона,
    из которого вычтено то, что играет динамик (музыка, голос Алисы) — см. aec.py."""

    def __init__(self, cancel_echo=True):
        from scipy.signal import firwin, lfilter
        # 48 -> 16 кГц: ФНЧ с состоянием между кусками (без щелчков на стыках) и каждый третий отсчёт.
        self.taps, self.lfilter = firwin(63, 7000, fs=48000), lfilter
        self.zi = np.zeros((len(self.taps) - 1, 1 + len(REFS)))
        self.phase = 0
        self.rest = b''
        self.aec = EchoCanceller(len(REFS)) if cancel_echo else None
        self.played = -MUSIC_HOLD     # когда динамик последний раз звучал (опора не пустая)
        self.levels = collections.deque(maxlen=400)   # (время, громкость опоры) по кускам — разбор обрывов

    @property
    def music(self):
        """Динамик звучит сейчас или звучал только что — музыка, голос Алисы."""
        return time.monotonic() - self.played < MUSIC_HOLD

    def feed(self, block):
        # read() без буфера отдаёт «сколько есть»: неполный кадр переносим, иначе каналы съедут.
        block = self.rest + block
        cut = len(block) // FRAME * FRAME
        block, self.rest = block[:cut], block[cut:]
        frames = np.frombuffer(block, '<i2').reshape(-1, CHANNELS)[:, [MIC, *REFS]].astype(np.float64)
        filtered, self.zi = self.lfilter(self.taps, 1.0, frames, axis=0, zi=self.zi)
        down = filtered[self.phase::3]
        self.phase = (self.phase - len(frames)) % 3
        level = (down[:, 1:] - down[:, 1:].mean(axis=0)).std() if len(down) else 0.0
        self.levels.append((time.monotonic(), level))
        if level > REF_ACTIVE:
            self.played = time.monotonic()
        mic = self.aec.process(down[:, 0], down[:, 1:].T) if self.aec else down[:, 0]
        return np.clip(mic, -32768, 32767).astype('<i2').tobytes()


def question_after_wake(text, heard):
    """Что сказано после «Джарвиса»: '' — только имя, None — имени не было."""
    words = re.sub(r'[^\w\s-]', ' ', text.lower()).split()   # на случай заглавных и знаков препинания
    # Последнее имя: в звук может попасть хвост прошлой фразы («…волну джарвис какая погода»).
    at = next((i for i in reversed(range(len(words))) if any(v in words[i] for v in WAKE)), None)
    if at is None:
        if not heard:
            return None
        # Грамматика имя услышала, а свободное распознавание исказило («тавис») — это первое слово.
        at = 0
    rest = ' '.join(words[at + 1:])
    return rest if len(rest) >= 3 else ''


class Session:
    """Подключения к одной колонке: туннель к Алисе, оболочка для звука и подсветки, поток микрофона."""

    def __init__(self, audio=True):
        self.tunnel = open_tunnel()
        self.alice = socket.create_connection(('127.0.0.1', LOCAL_PORT))
        self.alice_used = time.monotonic()
        self.shell = subprocess.Popen(SSH + ['sh'], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
        # --station: имя ловит сама колонка — поток микрофона не нужен.
        self.audio = subprocess.Popen(SSH + ['sh /data/opt/jarvis/pull.sh'], stdin=subprocess.DEVNULL,
                                      stdout=subprocess.PIPE, bufsize=0) if audio else None
        self.lock = threading.Lock()   # Gemini отвечает из своего потока — не перемешать кадры и команды
        self.busy = False              # Gemini ещё думает над прошлым вопросом
        # Под музыкой вопрос распознаёт Алиса: её события читаем отдельным подключением.
        self.capturing = False
        self.capture_deadline = 0.0
        self.on_question = None        # куда отдать перехваченный текст (ставит run)
        self.speaking = 0.0            # когда Джарвис начал говорить (0 — молчит): журнал событий Алисы
        self.events = None
        if audio:
            # Не в --station: события Алисы читает программа на колонке, а вторая подписка их у неё
            # отбирает (28.10: после запуска ПК колонка перестала получать распознанные вопросы).
            self.events = socket.create_connection(('127.0.0.1', LOCAL_PORT))
            self.events.sendall(alice_say.rpc_frame(20, alice_say.byte_field(1, b'AliceCapability')))
            threading.Thread(target=self._read_events, daemon=True).start()

    def capture(self):
        """Алиса приглушает музыку и слушает; её распознанный текст заберёт _read_events."""
        self.capturing, self.capture_deadline = True, time.monotonic() + CAPTURE_MAX
        self.listen()

    def _read_events(self):
        buf, text = b'', ''
        while True:
            try:
                chunk = self.events.recv(65536)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while len(buf) >= 4 and len(buf) >= 4 + int.from_bytes(buf[:4], 'little'):
                size = int.from_bytes(buf[:4], 'little')
                frame, buf = buf[4:4 + size], buf[4 + size:]
                event = alice_event(frame) if self.capturing or self.speaking else None
                if event is None:
                    continue
                if self.speaking and not self.capturing:
                    # Разбор обрывов речи Джарвиса: что делает Алиса, пока он говорит.
                    print(f'  [Алиса во время речи: событие {event[0]} «{event[1]}» '
                          f'{time.monotonic() - self.speaking:.1f} с от начала]', flush=True)
                    continue
                method, heard = event
                text = heard or text
                if method == RECOGNITION_COMPLETED:
                    # Сразу после распознавания: ответ Алисы приходит примерно через секунду.
                    self._alice(alice_method(CANCEL_DIALOG_AND_CLEAR_QUEUE) + alice_method(STOP_TTS))
                    self.capturing = False
                    if text and self.on_question:
                        self.on_question(text)
                    text = ''

    def _shell(self, line):
        with self.lock:
            self.shell.stdin.write((line + '\n').encode())
            self.shell.stdin.flush()

    def _alice(self, frame):
        # 28.10 в 19:05 вопрос ушёл Алисе в соединение, простоявшее 2,5 ч, — без ошибки, но и без ответа;
        # тот же текст в новом соединении она озвучивает. После простоя — новое соединение, при ошибке — повтор.
        with self.lock:
            for attempt in range(2):
                try:
                    if attempt or time.monotonic() - self.alice_used > ALICE_IDLE:
                        self.alice.close()
                        self.alice = socket.create_connection(('127.0.0.1', LOCAL_PORT), timeout=5)
                    self.alice.sendall(frame)
                    break
                except OSError as error:
                    if attempt:
                        print(f'  Нет связи с Алисой ({error})', flush=True)
            self.alice_used = time.monotonic()

    def reply(self, name):
        ms = DURATIONS.get(name, 1000)
        self._shell(f'(sh /data/opt/think/anim.sh /data/opt/think/speaking.bin {ms + 300} & '
                    f'{GST.format(name=name)} >/dev/null 2>&1; wait) &')
        return ms

    def think(self, on):
        # pkill в busybox колонки нет: гасим анимацию по сохранённому номеру процесса.
        if on:
            self._shell(f'sh /data/opt/think/anim.sh /data/opt/think/thinking.bin {THINK_MS} & '
                        'echo $! > /tmp/jarvis_think.pid')
        else:
            self._shell('kill $(cat /tmp/jarvis_think.pid) 2>/dev/null; '
                        'dd if=/dev/zero of=/dev/fb0 bs=356 count=1 2>/dev/null')

    def listen(self):
        self._alice(alice_say.rpc_frame(8, alice_say.integer(1, START_CONVERSATION)))

    def ask(self, text):
        self._alice(text_frame(text))

    def say(self, text):
        self._alice(alice_say.vocalize_frame(text))

    def speech(self, rate):
        return Speech(rate)

    def close(self):
        for proc in (self.audio, self.shell, self.tunnel):
            if proc:
                proc.terminate()
        self.alice.close()
        if self.events:
            self.events.close()


class Speech:
    """Голос Джарвиса одним потоком: сырой PCM по SSH прямо в gst колонки, с подсветкой «говорю».
    Предложения дописываются, пока звучат предыдущие, — без паузы на запуск проигрывателя между ними.
    Пишет отдельный поток: синтез следующего предложения не ждёт, пока колонка проиграет текущее."""

    def __init__(self, rate):
        # Для разбора обрывов речи: строки gst (ошибки, конец потока, смена состояния) с временем колонки
        # (/proc/uptime), время до и после, сколько байт дошло (dd) — разбирает report().
        command = ('U(){ cut -d" " -f1 /proc/uptime; }; echo "T0 $(U)"; '
                   f'sh /data/opt/think/anim.sh /data/opt/think/speaking.bin {SPEAK_MAX_MS} & A=$!; '
                   f'dd bs=4096 2>/tmp/jarvis_dd | ({STREAM_GST.format(rate=rate)}) 2>&1 | '   # скобки: в gst «cd … &&»
                   'while read -r l; do case "$l" in *ERROR*|*WARN*|*EOS*|*STATE_CHANGED*pipeline*) '
                   'echo "$(U) $l";; esac; done; echo "T1 $(U)"; echo "DD $(tr "\\n" " " </tmp/jarvis_dd)"; '
                   'kill $A 2>/dev/null; dd if=/dev/zero of=/dev/fb0 bs=356 count=1 2>/dev/null')
        self.rate = rate
        self.seconds = 0.0                   # сколько звука отдано колонке
        self.started = time.monotonic()
        self.proc = subprocess.Popen(SSH + [command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
        self.queue = queue.Queue()
        self.failed = False
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.writer.start()

    def add(self, pcm):
        self.seconds += len(pcm) / 2 / self.rate
        self.queue.put(pcm)

    def _write(self):
        try:
            while (pcm := self.queue.get()) is not None:
                self.proc.stdin.write(pcm)
                self.proc.stdin.flush()
            self.proc.stdin.close()
        except OSError:   # SSH оборвался — остаток ответа прочитает Алиса
            self.failed = True

    def close(self):
        """Дождаться, пока колонка договорит."""
        self.queue.put(bytes(int(TAIL_SILENCE * self.rate) * 2))   # в self.seconds не входит: это не речь
        self.queue.put(None)
        self.writer.join(SPEAK_MAX_MS / 1000)
        try:
            log = self.proc.communicate(timeout=SPEAK_MAX_MS / 1000)[0].decode(errors='replace')
        except subprocess.TimeoutExpired:
            self.proc.kill()
            log = 'тайм-аут'
        spent = time.monotonic() - self.started
        print('  ' + self.report(log, spent), flush=True)
        if self.failed or 'ERROR' in log:
            print(f'  gst: {log.strip()[-400:]}', flush=True)

    def report(self, log, spent):
        """Журнал колонки -> строка: когда gst заиграл и закончил по часам колонки, сколько байт дошло."""
        times = {}
        for line in log.splitlines():
            stamp, _, text = line.partition(' ')
            if stamp in ('T0', 'T1'):
                times[stamp] = float(text)
            elif re.match(r'^\d+\.\d+$', stamp):
                if 'to GST_STATE_PLAYING' in text and 'play' not in times:
                    times['play'] = float(stamp)
                if 'EOS' in text and 'eos' not in times:
                    times['eos'] = float(stamp)
        got = re.search(r'(\d+)\+(\d+) records in', log)   # busybox dd: полных блоков по 4096 + неполных
        sent = int(self.seconds * self.rate * 2)
        if 'T0' not in times or 'T1' not in times:
            return f'Колонка: нет отметок времени (проигрыш {spent:.1f} с, звука {self.seconds:.1f} с)'
        t0, nan = times['T0'], float('nan')
        played = times.get('eos', nan) - times.get('play', nan)
        blocks = f'{got.group(1)}+{got.group(2)} блоков ≈ {int(got.group(1)) * 4096} байт' if got else '?'
        return (f'Колонка: gst заиграл через {times.get("play", nan) - t0:.2f} с, играл до конца потока '
                f'{played:.2f} с при звуке {self.seconds:.2f} с (недостача {self.seconds - played:+.2f} с), '
                f'вышел через {times["T1"] - t0:.2f} с; дошло {blocks} из {sent}')


def pcm_of(wav):
    """WAV от сервера голоса -> (частота, сырой PCM S16LE моно)."""
    with wave.open(io.BytesIO(wav)) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise wave.Error('ждали 16 бит моно')
        return w.getframerate(), w.readframes(w.getnframes())


def alice_event(frame):
    """Кадр канала Алисы -> (номер события, распознанный текст или '') или None.

    Путь найден наблюдением (alice_watch.py): событие слушателя — поле 18 конверта 252, номер — его
    поле 1, распознанный текст растёт в состоянии 2 -> 15 -> 1 («Какая» -> «Какая сегодня погода»),
    конец распознавания — событие RECOGNITION_COMPLETED."""
    try:
        listener = alice_say.fields(alice_say.fields(alice_say.fields(frame).get(252, b'')).get(18, b''))
        heard = alice_say.fields(alice_say.fields(listener.get(2, b'')).get(15, b'')).get(1, b'')
        return listener.get(1, 0), heard.decode('utf-8') if isinstance(heard, bytes) else ''
    except (ValueError, UnicodeDecodeError):
        return None


def jarvis_voice(text):
    """Текст -> WAV голосом Джарвиса или None, если сервер синтеза не запущен или ошибся."""
    request = urllib.request.Request(TTS_URL, data=text.encode('utf-8'), method='POST')
    try:
        with urllib.request.urlopen(request, timeout=TTS_TIMEOUT) as response:
            return response.read()
    except OSError as error:   # нет сервера, тайм-аут, HTTP-ошибка
        print(f'  Голос Джарвиса недоступен ({str(error)[:60]}) -> читает Алиса', flush=True)
        return None


BUSY_PHRASE = 'Сэр, серверы Google перегружены, передаю вопрос Алисе.'
spoken_cache = {}           # фраза -> WAV: служебные фразы синтезируем один раз


def jarvis_say(session, text):
    """Короткая фраза голосом Джарвиса; возвращает, когда колонка договорит. False — голос недоступен."""
    wav = spoken_cache.get(text) or jarvis_voice(text)
    try:
        rate, pcm = pcm_of(wav) if wav else (None, None)
    except (wave.Error, EOFError):
        return False
    if pcm is None:
        return False
    spoken_cache[text] = wav
    speech = session.speech(rate)
    session.speaking = time.monotonic()
    speech.add(pcm)
    speech.close()
    session.speaking = 0.0
    return True


def answer(session, brain, question, audio):
    """Звук вопроса -> Gemini; его ответ колонка читает голосом Джарвиса (F5-TTS), команду выполняет Алиса.
    Ответ идёт по предложениям: первое звучит, пока Gemini дописывает и сервер синтезирует остальные.
    Gemini недоступен — Джарвис говорит об этом, затем Алисе уходит текст Vosk (question).
    Голос Джарвиса недоступен — ответ (или его остаток) читает Алиса."""
    started = time.monotonic()
    session.think(True)
    thinking = True
    if audio:   # для разбора ослышек: что именно услышал Gemini
        with wave.open(str(LAST_QUESTION), 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(audio)

    def stop_thinking():
        nonlocal thinking
        if thinking:
            session.think(False)
            thinking = False

    speech, said, unsaid, kind = None, [], [], None
    try:
        for event, value in brain.ask_stream(question, audio):
            if event == 'heard':
                print(f'  Gemini услышал: «{value}»', flush=True)
                continue
            kind = event
            if event == 'alice':
                stop_thinking()
                print(f'  Gemini -> Алисе: «{value}» ({time.monotonic() - started:.1f} с)', flush=True)
                session.ask(value)
                continue
            # Кусок речи: голосом Джарвиса, пока голос работает; после первого сбоя — копим для Алисы.
            pcm = None
            if not unsaid and not (speech and speech.failed):
                wav = jarvis_voice(value)     # пока синтез, на колонке анимация «думаю»
                try:
                    rate, pcm = pcm_of(wav) if wav else (None, None)
                except (wave.Error, EOFError) as error:
                    print(f'  Сервер голоса прислал не тот звук ({error}) -> читает Алиса', flush=True)
            if pcm is None:
                unsaid.append(value)
                continue
            if speech is None:
                stop_thinking()
                speech = session.speech(rate)
                session.speaking = time.monotonic()
                print(f'  Джарвис заговорил через {time.monotonic() - started:.1f} с', flush=True)
            speech.add(pcm)
            said.append(value)
    except Exception as error:   # сеть, квота, перегрузка всех моделей
        if kind is None:
            print(f'  Gemini недоступен ({str(error)[:80]}) -> вопрос Алисе', flush=True)
            kind = 'alice'
            stop_thinking()
            jarvis_say(session, BUSY_PHRASE)    # иначе непонятно, почему отвечает Алиса (или молчит)
            session.ask(question)
        else:
            print(f'  Gemini оборвал ответ ({str(error)[:80]})', flush=True)
    finally:
        stop_thinking()
        session.busy = False
    spent = time.monotonic() - started
    if kind is None:
        print(f'  Gemini промолчал ({spent:.1f} с)', flush=True)
    if said:
        print(f'  Gemini ({spent:.1f} с, {brain.model}, Джарвис): {" ".join(said)}', flush=True)
    if speech:
        speech.close()          # Алиса не должна заговорить поверх Джарвиса
        session.speaking = 0.0
        decoder = getattr(session, 'decoder', None)
        if decoder:
            # Что реально звучало из динамика (опорные каналы; поток с колонки отстаёт ~1 с), по 0,25 с
            # от начала речи: «.» — тишина, цифра — громкость. Обрыв речи виден как точки в середине.
            bins = {}
            for at, level in decoder.levels:
                if at >= speech.started:
                    slot = int((at - speech.started) / 0.25)
                    bins[slot] = max(bins.get(slot, 0.0), level)
            marks = ''.join('.' if bins.get(i, 0.0) < REF_ACTIVE
                            else str(min(9, int(np.log2(bins[i] / REF_ACTIVE)) + 1))
                            for i in range(max(bins, default=-1) + 1))
            print(f'  Динамик за ответ ({speech.seconds:.1f} с звука, по 0,25 с): {marks}', flush=True)
    if unsaid:
        print(f'  Gemini ({spent:.1f} с, Алиса): {" ".join(unsaid)}', flush=True)
        session.say(' '.join(unsaid))


def run(session, wake, full, brain):
    decoder = MicDecoder()
    session.decoder = decoder      # answer() печатает по нему, что реально звучало из динамика
    quiet_until = 0.0
    heard = False             # грамматика услышала имя в текущей фразе
    utter = bytearray()       # звук текущей фразы (после вычитания музыки) — для Gemini
    got, tick = 0, time.monotonic()

    def captured(text):       # Алиса распознала вопрос, заданный под музыкой
        print(f'  Алиса распознала: «{text}» -> Gemini', flush=True)
        if not session.busy:
            session.busy = True
            threading.Thread(target=answer, args=(session, brain, text, None), daemon=True).start()

    session.on_question = captured
    while True:
        block = session.audio.stdout.read(CHUNK)
        if not block:
            return
        pcm = decoder.feed(block)
        if ASK_FILE.exists() and not session.busy:
            # Проверочный вход: вопрос текстом из файла — тот же путь ответа, что и голосом.
            text = ASK_FILE.read_text(encoding='utf-8').strip()
            ASK_FILE.unlink()
            if text.startswith('!'):
                print(f'{time.strftime("%H:%M:%S")} из файла сразу Алисе: «{text[1:].strip()}»', flush=True)
                session.ask(text[1:].strip())
            elif text:
                print(f'{time.strftime("%H:%M:%S")} вопрос из файла: «{text}»', flush=True)
                session.busy = True
                threading.Thread(target=answer, args=(session, brain, text, None), daemon=True).start()
        utter += pcm
        del utter[:-UTTER_BYTES]
        if DEBUG:
            got += len(block)
            if time.monotonic() - tick >= 2:
                level = int(np.abs(np.frombuffer(pcm, '<i2')).max()) if pcm else 0
                print(f'  поток {got / (time.monotonic() - tick):.0f} Б/с, пик {level}, '
                      f'partial={json.loads(full.PartialResult()).get("partial", "")!r}', flush=True)
                got, tick = 0, time.monotonic()
        if session.capturing:
            # Вопрос сейчас слушает Алиса — свои распознаватели только прокручиваем.
            if time.monotonic() > session.capture_deadline:
                print('  Алиса так и не распознала вопрос', flush=True)
                session.capturing = False
            wake.AcceptWaveform(pcm)
            full.AcceptWaveform(pcm)
            continue
        wake_final = wake.AcceptWaveform(pcm)
        if wake_final:
            heard |= 'джарвис' in json.loads(wake.Result()).get('text', '')
        ended = full.AcceptWaveform(pcm)
        if decoder.music and not wake_final and not ended:
            # Под музыкой не ждём конца фразы — смотрим промежуточные результаты. Грамматика из двух
            # слов мигает «джарвисом» и на других словах, поэтому имя должен видеть и свободный.
            if 'джарвис' in json.loads(wake.PartialResult()).get('partial', '') and \
                    any(v in json.loads(full.PartialResult()).get('partial', '') for v in WAKE):
                heard = True
        if heard and decoder.music and time.monotonic() >= quiet_until and not session.busy:
            # Под музыкой конца фразы не ждём: Алиса сразу приглушает музыку, как на своё имя,
            # и распознаёт вопрос облаком; текст заберёт Session._read_events и отдаст Gemini.
            print(f'{time.strftime("%H:%M:%S")} «джарвис» под музыкой -> Алиса приглушает и слушает',
                  flush=True)
            session.capture()
            heard, utter = False, bytearray()
            quiet_until = time.monotonic() + COOLDOWN
            wake.Reset()
            full.Reset()
            continue
        if not ended:
            continue
        result = json.loads(full.Result())
        text = result.get('text', '')
        question = question_after_wake(text, heard)
        heard = False
        audio, utter = bytes(utter), bytearray()
        if DEBUG and text:
            print(f'  фраза: {text!r} -> {question!r}', flush=True)
        if question is None or time.monotonic() < quiet_until:
            continue
        stamp = time.strftime('%H:%M:%S')
        if question in QUICK:
            print(f'{stamp} «джарвис, {question}» -> сразу Алисе', flush=True)
            session.ask(question)
        elif question:
            if session.busy:
                print(f'{stamp} «джарвис, {question}» — пропущен: Gemini ещё отвечает на прошлый', flush=True)
                continue
            # Текст Vosk здесь черновой (только для журнала и запасного пути) — Gemini слушает звук сам.
            print(f'{stamp} «джарвис, …» (Vosk: «{question}») -> звук в Gemini', flush=True)
            session.busy = True
            threading.Thread(target=answer, args=(session, brain, question, audio), daemon=True).start()
        else:
            name = random.choice(['reply1', 'reply2', 'reply3'])
            print(f'{stamp} «джарвис» -> {name}, затем Алиса слушает вопрос для Gemini', flush=True)
            time.sleep(session.reply(name) / 1000 + 0.2)
            session.capture()
        quiet_until = time.monotonic() + COOLDOWN
        wake.Reset()
        full.Reset()


# --station: «Джарвис» ловит сама колонка (jarvis-station: программа jarvis в режиме pc), Алиса распознаёт
# вопрос, колонка кладёт его строкой в канал /tmp/jarvis-q. Читатель держит канал открытым (3<>), пока жив:
# пока его нет, колонка отдаёт вопрос Алисе. Раз в 5 с — PING: запись в оборванный SSH роняет читателя,
# и канал освобождается. Без канала (программа на колонке не запущена) читатель выходит — `<>` создал бы
# вместо канала обычный файл.
STATION_READER = ('[ -p /tmp/jarvis-q ] || { echo NOFIFO; exit 3; }; exec 3<>/tmp/jarvis-q; '
                  'while :; do if read -r -t 5 q <&3; then echo "Q $q"; else echo PING; fi || exit 0; done')


def run_station(session, brain):
    reader = subprocess.Popen(SSH + [STATION_READER], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    try:
        for raw in reader.stdout:
            line = raw.decode('utf-8', errors='replace').strip()
            if line == 'NOFIFO':
                print('На колонке не запущен jarvis в режиме pc (нет /tmp/jarvis-q).', flush=True)
            question = line[2:].strip() if line.startswith('Q ') else (check_ask_file(session) or '')
            if not question:
                continue
            stamp = time.strftime('%H:%M:%S')
            if session.busy:
                print(f'{stamp} «{question}» — пропущен: Gemini ещё отвечает на прошлый', flush=True)
                continue
            print(f'{stamp} колонка: «{question}» -> Gemini', flush=True)
            session.busy = True
            threading.Thread(target=answer, args=(session, brain, question, None), daemon=True).start()
    finally:
        reader.terminate()


def check_ask_file(session):
    """Проверочный вопрос из ASK_FILE (как в run): текст для Gemini или '' (в том числе «!…» — сразу Алисе)."""
    if not ASK_FILE.exists() or session.busy:
        return ''
    text = ASK_FILE.read_text(encoding='utf-8').strip()
    ASK_FILE.unlink()
    if text.startswith('!'):
        session.ask(text[1:].strip())
        return ''
    return text


def main_station():
    brain = Brain()
    print('Джарвис: имя ловит колонка, ответы — Gemini и голос с этого ПК. Ctrl+C — выход.', flush=True)
    try:
        while True:
            session = None
            try:
                session = Session(audio=False)
                run_station(session, brain)
                print('Связь с колонкой прервалась, переподключаюсь.', flush=True)
            except (OSError, RuntimeError) as error:
                print(f'Нет связи с колонкой ({error}), повтор через {RECONNECT:g} с.', flush=True)
            finally:
                if session:
                    session.close()
            time.sleep(RECONNECT)
    except KeyboardInterrupt:
        print('Выход.')


def main():
    import vosk
    vosk.SetLogLevel(-1)
    model = vosk.Model(str(MODEL))
    wake = vosk.KaldiRecognizer(model, 16000, GRAMMAR)
    full = vosk.KaldiRecognizer(model, 16000)
    brain = Brain()           # одна на все переподключения: память разговора не теряется
    print('Джарвис слушает микрофон колонки. Скажите «Джарвис» или «Джарвис, вопрос». Ctrl+C — выход.',
          flush=True)
    try:
        while True:
            session = None
            try:
                session = Session()
                run(session, wake, full, brain)
                print('Поток с колонки прервался, переподключаюсь.', flush=True)
            except (OSError, RuntimeError) as error:
                print(f'Нет связи с колонкой ({error}), повтор через {RECONNECT:g} с.', flush=True)
            finally:
                if session:
                    session.close()
            wake.Reset()
            full.Reset()
            time.sleep(RECONNECT)
    except KeyboardInterrupt:
        print('Выход.')


if __name__ == '__main__':
    main_station() if '--station' in sys.argv[1:] else main()
