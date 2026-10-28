"""Озвучить текст голосом Алисы на Станции через SSH-туннель (без socat на колонке).

    py say.py "Текст"                  # сказать сразу
    py say.py --delay 4 "Текст"        # подождать 4 секунды и сказать
    py say.py --think 2500 "Текст"     # 2,5 с анимации «думаю» на панели, потом сказать

Запросы собирает alice_say.py (contrib/alice-say.py из yandex-station-extended).
"""
import argparse
import asyncio
import socket
import subprocess
import sys
import time

import alice_say
from jarvis_settings import HOST, SSH_PORT

LOCAL_PORT = 19875


def think(ms):
    # Кадры штатной alice_thinking.led, проигрывает /data/opt/think/think.sh на колонке.
    subprocess.run(['ssh', '-p', SSH_PORT, '-o', 'BatchMode=yes', f'root@{HOST}',
                    f'sh /data/opt/think/think.sh {int(ms)}'], stdin=subprocess.DEVNULL, check=True)


def open_tunnel(port=LOCAL_PORT):
    proc = subprocess.Popen(
        ['ssh', '-N', '-p', SSH_PORT, '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
         '-o', 'ConnectTimeout=6', '-L', f'{port}:127.0.0.1:9875', f'root@{HOST}'],
        stdin=subprocess.DEVNULL)
    for _ in range(50):
        if proc.poll() is not None:
            raise RuntimeError(f'SSH не поднял туннель: проверьте ssh -p {SSH_PORT} root@{HOST}')
        try:
            socket.create_connection(('127.0.0.1', port), timeout=0.2).close()
            return proc
        except OSError:
            time.sleep(0.1)
    proc.terminate()
    raise RuntimeError('Туннель не открылся за 5 секунд.')


async def speak(text, timeout):
    reader, writer = await asyncio.open_connection('127.0.0.1', LOCAL_PORT)
    started = False
    try:
        async with asyncio.timeout(timeout):
            subscribe = alice_say.rpc_frame(20, alice_say.byte_field(1, b'AliceCapability'))
            writer.write(subscribe + alice_say.vocalize_frame(text))
            await writer.drain()
            while True:
                size = int.from_bytes(await reader.readexactly(4), 'little')
                if not 0 < size <= 512 * 1024:
                    raise ValueError('Некорректный размер ответа колонки.')
                envelope = alice_say.fields(await reader.readexactly(size))
                if envelope.get(253) != 3:
                    continue
                remote = alice_say.fields(envelope.get(252, b''))
                if 18 not in remote:
                    continue
                event = alice_say.fields(remote[18]).get(1)
                if event == 6:
                    started = True
                    print('Озвучка началась.', flush=True)
                elif event == 1 and started:
                    print('Готово.', flush=True)
                    return
                elif event == 4:
                    raise RuntimeError('Алиса сообщила об ошибке озвучки.')
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description='Сказать текст голосом Алисы.')
    parser.add_argument('text')
    parser.add_argument('--delay', type=float, default=0, help='Пауза перед озвучкой, секунды')
    parser.add_argument('--think', type=int, default=0, help='Анимация «думаю» перед ответом, мс')
    # Станция Midi не присылает событий озвучки, поэтому долго ждать их незачем.
    parser.add_argument('--timeout', type=float, default=5)
    args = parser.parse_args()

    tunnel = open_tunnel()
    try:
        if args.delay > 0:
            print(f'Скажу через {args.delay:g} с...', flush=True)
            time.sleep(args.delay)
        if args.think > 0:
            think(args.think)
        asyncio.run(speak(args.text, args.timeout))
    except TimeoutError:
        print('Отправлено.', flush=True)
    finally:
        tunnel.terminate()
    return 0


if __name__ == '__main__':
    sys.exit(main())
