"""Отклик на «Алиса, Джарвис…»: отменить ответ Алисы и ответить фразой Джарвиса с золотой подсветкой.

    py jarvis_listen.py            # работает, пока не нажать Ctrl+C

Фразы — набор jarvis-og из github.com/Priler/jarvis (CC BY-NC-SA 4.0), лежат на колонке в /data/opt/jarvis/og.
"""
import json
import pathlib
import random
import socket
import subprocess
import time

import alice_say
from say import LOCAL_PORT, open_tunnel
from jarvis import SSH

HERE = pathlib.Path(__file__).parent
DURATIONS = json.loads((HERE / 'jarvis-og-wav' / 'durations.json').read_text(encoding='utf-8'))
WAKE = ('джарвис', 'джарвиз', 'джервис', 'жарвис', 'джарвес', 'jarvis')  # как его слышит распознавание
COOLDOWN = 2.5  # после ответа: хвост той же фразы ещё приходит кадрами — его пропускаем
QUIET, MAX_WAIT = 0.6, 2.5  # фраза «дописана», если не росла QUIET с; дольше MAX_WAIT не ждём
# Номера из quasar-protos: yandex_io/protos/remoting.proto, AliceCapabilityMethod.
CANCEL_DIALOG_AND_CLEAR_QUEUE, STOP_TTS = 6, 14
REPLIES = [  # первое совпадение по словам фразы, иначе случайный отклик
    (('спасибо', 'благодар'), ['thanks']),
    (('отключ', 'выключ', 'спать'), ['off']),
    (('игров',), ['game_mode']),
    (('дурак', 'тупой', 'глуп'), ['stupid']),
    (('сделай', 'включи', 'запусти'), ['ok1', 'ok2', 'ok3', 'ok4']),
]
GST = ('cd /system/vendor/quasar && LD_LIBRARY_PATH=/system/vendor/quasar:libs ./maind --gst-launch '
       '"filesrc location=/data/opt/jarvis/og/{name}.wav ! decodebin ! audioconvert ! audioresample ! alsasink"')


def strings(data, depth=0):
    """Все строки внутри protobuf-кадра (распознанный текст лежит на разной глубине)."""
    try:
        parsed = alice_say.fields(data)
    except ValueError:
        return
    for value in parsed.values():
        if isinstance(value, bytes):
            try:
                text = value.decode('utf-8')
                if text.isprintable():
                    yield text
                    continue
            except UnicodeDecodeError:
                pass
            if depth < 6:
                yield from strings(value, depth + 1)


def pick(phrase):
    for words, names in REPLIES:
        if any(w in phrase for w in words):
            return random.choice(names)
    return random.choice(['reply1', 'reply2', 'reply3'])


def alice_method(method):
    # silent=true (поле 7): отменить без звукового сигнала ошибки.
    return alice_say.rpc_frame(8, alice_say.integer(1, method) + alice_say.integer(7, 1))


def main():
    tunnel = open_tunnel()
    sock = socket.create_connection(('127.0.0.1', LOCAL_PORT))
    sock.sendall(alice_say.rpc_frame(20, alice_say.byte_field(1, b'AliceCapability')))
    # Постоянная SSH-сессия: команды на колонку уходят без задержки на подключение.
    # Байтовый режим: в текстовом Windows дописывает \r, и sh на колонке ругается «: not found».
    shell = subprocess.Popen(SSH + ['sh'], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
    print('Джарвис слушает. Скажите «Алиса, Джарвис…». Ctrl+C — выход.', flush=True)
    sock.settimeout(0.1)
    buf, last, seen = b'', -COOLDOWN, set()
    phrase, started, grew = None, 0.0, 0.0
    try:
        while True:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                chunk = b''
            else:
                if not chunk:
                    print('Канал колонки закрылся.')
                    break
            buf += chunk
            while len(buf) >= 4:
                size = int.from_bytes(buf[:4], 'little')
                if len(buf) < 4 + size:
                    break
                frame, buf = buf[4:4 + size], buf[4 + size:]
                heard = [t.lower() for t in strings(frame) if any('а' <= c <= 'я' for c in t.lower())]
                for t in heard:  # для отладки: что распознала Алиса
                    if len(t) < 100 and t not in seen:
                        seen.add(t)
                        print(f'{time.strftime("%H:%M:%S")}   слышу: {t}', flush=True)
                texts = [t for t in heard if any(w in t for w in WAKE)]
                if not texts:
                    continue
                now = time.monotonic()
                if phrase is not None:  # дописываем фразу, пока распознавание её растит
                    longest = max(texts, key=len)
                    if len(longest) > len(phrase):
                        phrase, grew = longest, now
                    continue
                if now < last + COOLDOWN:
                    continue
                phrase, started, grew = max(texts, key=len), now, now
            # Отмена сразу обрывает и распознавание, поэтому гасим Алису, только когда фраза
            # перестала расти (до её ответа после распознавания около 2 с).
            now = time.monotonic()
            if phrase is not None and (now - grew >= QUIET or now - started >= MAX_WAIT):
                sock.sendall(alice_method(CANCEL_DIALOG_AND_CLEAR_QUEUE))
                sock.sendall(alice_method(STOP_TTS))
                name = pick(phrase)
                print(f'{time.strftime("%H:%M:%S")} «{phrase}» -> {name}', flush=True)
                ms = DURATIONS.get(name, 1500)
                shell.stdin.write(
                    (f'(sh /data/opt/think/anim.sh /data/opt/think/speaking.bin {ms + 300} & '
                     f'{GST.format(name=name)} >/dev/null 2>&1; wait) &\n').encode())
                shell.stdin.flush()
                phrase, last = None, time.monotonic()
    except KeyboardInterrupt:
        print('Выход.')
    finally:
        sock.close()
        shell.terminate()
        tunnel.terminate()


if __name__ == '__main__':
    main()
