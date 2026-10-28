"""Посмотреть, что раздаёт локальный порт колонки (по умолчанию yio_audio 9850): ищем куски звука audio_inputs.

    py audio_probe.py [порт] [секунды]
"""
import collections
import socket
import subprocess
import sys
import time

import alice_say
from jarvis_settings import HOST, SSH_PORT


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9850
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 8
    local = 20000 + port % 10000
    tunnel = subprocess.Popen(['ssh', '-N', '-p', SSH_PORT, '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
                               '-L', f'{local}:127.0.0.1:{port}', f'root@{HOST}'], stdin=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                sock = socket.create_connection(('127.0.0.1', local), timeout=0.2)
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise SystemExit('туннель не поднялся')
        sock.settimeout(1)
        buf, end, frames, fields_seen = b'', time.time() + seconds, 0, collections.Counter()
        channels, audio_bytes = collections.Counter(), 0
        while time.time() < end:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                print('порт закрыл соединение')
                break
            buf += chunk
            while len(buf) >= 4:
                size = int.from_bytes(buf[:4], 'little')
                if len(buf) < 4 + size:
                    break
                frame, buf = buf[4:4 + size], buf[4 + size:]
                frames += 1
                try:
                    top = alice_say.fields(frame)
                except ValueError:
                    fields_seen['не protobuf'] += 1
                    continue
                fields_seen.update(top.keys())
                if 108 in top:  # audio_inputs (repeated: fields() оставляет последний элемент)
                    ai = alice_say.fields(top[108])
                    channels[ai.get(3, b'?').decode('utf-8', 'replace')] += 1
                    audio_bytes += len(ai.get(1, b''))
        print(f'кадров: {frames}, поля верхнего уровня: {dict(fields_seen)}')
        print(f'каналы звука: {dict(channels)}, байт звука (последние элементы): {audio_bytes}')
    finally:
        tunnel.terminate()


if __name__ == '__main__':
    main()
