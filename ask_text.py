"""Задать Алисе вопрос текстом (как из приложения): она отвечает своим голосом, без распознавания речи.

    py ask_text.py "как дела"
    set ASK_DUMP=1   # печатать строки из кадров состояния
"""
import argparse
import asyncio
import json
import os
import re
import uuid

from alice_say import byte_field, fields, integer, rpc_frame
from say import open_tunnel

DUMP = bool(os.environ.get('ASK_DUMP'))
START_REQUEST, TEXT_INPUT = 0, 1   # remoting.proto AliceCapabilityMethod, VinsRequest.RequestType
PRINTABLE = re.compile(rb'(?:[\x20-\x7e]|[\xd0\xd1][\x80-\xbf]){4,}')


def text_frame(text):
    event = json.dumps({'type': 'text_input', 'text': text}, ensure_ascii=False)
    vins = (byte_field(1, str(uuid.uuid4()).encode()) + byte_field(2, event.encode('utf-8'))
            + integer(4, 0) + integer(12, 1) + integer(18, TEXT_INPUT))  # is_silent=0, voice_session=1
    return rpc_frame(8, integer(1, START_REQUEST) + byte_field(2, vins) + integer(3, 1))


async def ask(text, seconds, port):
    tunnel = open_tunnel(port)
    reader, writer = await asyncio.open_connection('127.0.0.1', port)
    writer.write(rpc_frame(20, byte_field(1, b'AliceCapability')) + text_frame(text))
    await writer.drain()
    try:
        async with asyncio.timeout(seconds):
            while True:
                size = int.from_bytes(await reader.readexactly(4), 'little')
                envelope = fields(await reader.readexactly(size))
                if envelope.get(253) != 3:
                    continue
                remote = fields(envelope.get(252, b''))
                event = fields(remote[18]).get(1) if 18 in remote else None
                print('событие', event, 'поля', sorted(remote), flush=True)
                if DUMP:
                    found = PRINTABLE.findall(envelope.get(252, b''))
                    print('   ', [f.decode('utf-8', 'replace') for f in found][:12], flush=True)
    except TimeoutError:
        pass
    finally:
        writer.close()
        tunnel.terminate()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('text')
    p.add_argument('--port', type=int, default=19876)
    p.add_argument('--seconds', type=float, default=12)
    a = p.parse_args()
    asyncio.run(ask(a.text, a.seconds, a.port))
