"""Показать, что Алиса присылает по локальному каналу 9875 (состояния и распознанный текст).

    py alice_probe.py [секунды]
"""
import socket
import sys
import time

import alice_say
from say import LOCAL_PORT, open_tunnel


def dump(data, depth=0):
    """Рекурсивно разобрать protobuf и показать номера полей, числа и строки."""
    try:
        parsed = alice_say.fields(data)
    except ValueError:
        return False
    for key, value in parsed.items():
        if isinstance(value, bytes):
            text = None
            try:
                text = value.decode('utf-8')
            except UnicodeDecodeError:
                pass
            print('  ' * depth + f'{key}: bytes[{len(value)}]' + (f' "{text[:80]}"' if text and text.isprintable() else ''))
            if depth < 6 and not (text and text.isprintable()):
                dump(value, depth + 1)
        else:
            print('  ' * depth + f'{key}: {value}')
    return True


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60
    tunnel = open_tunnel()
    sock = socket.create_connection(('127.0.0.1', LOCAL_PORT))
    sock.sendall(alice_say.rpc_frame(20, alice_say.byte_field(1, b'AliceCapability')))
    sock.settimeout(1)
    end, buf = time.time() + seconds, b''
    print(f'Слушаю {seconds:g} с...', flush=True)
    try:
        while time.time() < end:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                print('Канал закрыт.')
                break
            buf += chunk
            while len(buf) >= 4:
                size = int.from_bytes(buf[:4], 'little')
                if len(buf) < 4 + size:
                    break
                frame, buf = buf[4:4 + size], buf[4 + size:]
                print(f'--- {time.strftime("%H:%M:%S")} кадр {size} байт', flush=True)
                dump(frame)
                sys.stdout.flush()
    finally:
        sock.close()
        tunnel.terminate()


if __name__ == '__main__':
    main()
