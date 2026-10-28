"""Наблюдатель за Алисой: события её канала и строки в них с путём полей. Ничего не отправляет.

    py alice_watch.py [секунды]
Нужен, чтобы найти, где лежит распознанный голосом текст и когда распознавание закончено.
"""
import asyncio
import sys
import time

from alice_say import byte_field, fields, rpc_frame
from say import open_tunnel

PORT = 19888
METHODS = {0: 'STATE_CHANGED', 1: 'TTS_COMPLETED', 2: 'ON_ALICE_REQUEST', 3: 'ON_ALICE_RESPONSE',
           4: 'ON_ALICE_ERROR', 6: 'TTS_STARTED', 7: 'RECOGNITION_COMPLETED', 8: 'ON_ALICE_EVENT'}


def russian(data, path, depth=0):
    try:
        parsed = fields(data)
    except Exception:
        return
    for key, value in parsed.items():
        if not isinstance(value, bytes):
            continue
        try:
            text = value.decode('utf-8')
            if text.isprintable() and any('а' <= c <= 'я' for c in text.lower()) and len(text) < 200:
                yield path + [key], text
                continue
        except UnicodeDecodeError:
            pass
        if depth < 10:
            yield from russian(value, path + [key], depth + 1)


async def main(seconds):
    tunnel = open_tunnel(PORT)
    reader, writer = await asyncio.open_connection('127.0.0.1', PORT)
    writer.write(rpc_frame(20, byte_field(1, b'AliceCapability')))
    await writer.drain()
    print('Наблюдаю за Алисой. Скажите ей что-нибудь.', flush=True)
    last = None
    try:
        async with asyncio.timeout(seconds):
            while True:
                size = int.from_bytes(await reader.readexactly(4), 'little')
                envelope = fields(await reader.readexactly(size))
                remote = fields(envelope.get(252, b''))
                if 18 not in remote:
                    continue
                listener = fields(remote[18])
                method = METHODS.get(listener.get(1, 0), listener.get(1, 0))
                found = sorted(set((tuple(p), t) for p, t in russian(remote[18], [18])))
                line = f'{method}: ' + '; '.join(f'{list(p)}={t!r}' for p, t in found)
                if line != last:
                    print(time.strftime('%H:%M:%S'), line, flush=True)
                    last = line
    except TimeoutError:
        pass
    finally:
        writer.close()
        tunnel.terminate()


asyncio.run(main(float(sys.argv[1]) if len(sys.argv) > 1 else 180))
