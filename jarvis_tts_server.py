"""Сервер синтеза голоса Джарвиса: держит F5-TTS в памяти, отдаёт WAV по HTTP только на 127.0.0.1.

    %USERPROFILE%\\Downloads\\f5-tts\\venv\\Scripts\\python.exe jarvis_tts_server.py

POST /tts, тело — текст в UTF-8 -> audio/wav (24 кГц, 16 бит, моно). GET /health -> 200, когда модель готова.
Слушатель (jarvis_wake.py) работает на другом питоне без torch, поэтому синтез — отдельным процессом.
Запросы обрабатываются по одному: модель на видеокарте одна.
"""
import io
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlsplit

import soundfile as sf

from jarvis_voice import Voice

HOST, PORT = '127.0.0.1', 19890
MAX_TEXT = 1000     # Gemini отвечает 1–3 предложениями; длиннее — ошибка, а не долгий синтез


class Handler(BaseHTTPRequestHandler):
    voice = None

    def do_GET(self):
        self._send(200 if self.path == '/health' else 404, b'ok', 'text/plain')

    def do_POST(self):
        url = urlsplit(self.path)
        if url.path != '/tts':
            return self._send(404, b'', 'text/plain')
        try:
            text = self.rfile.read(int(self.headers.get('Content-Length', 0))).decode('utf-8').strip()
        except (ValueError, UnicodeDecodeError):
            return self._send(400, 'текст должен быть в UTF-8'.encode(), 'text/plain; charset=utf-8')
        if not text or len(text) > MAX_TEXT:
            return self._send(400, 'пустой или слишком длинный текст'.encode(), 'text/plain; charset=utf-8')
        nfe = parse_qs(url.query).get('nfe', [''])[0]   # /tts?nfe=16 — сравнить скорость и качество
        t = time.perf_counter()
        wav, rate = self.voice.synth(text, int(nfe) if nfe.isdigit() else None)
        buf = io.BytesIO()
        sf.write(buf, wav, rate, format='WAV', subtype='PCM_16')
        print(f'{len(wav) / rate:.1f} с звука за {time.perf_counter() - t:.2f} с: {text[:80]}', flush=True)
        self._send(200, buf.getvalue(), 'audio/wav')

    def _send(self, code, body, kind):
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def main():
    t = time.perf_counter()
    Handler.voice = Voice()
    Handler.voice.synth('Готов.')      # прогрев: первый синтез на видеокарте заметно медленнее
    print(f'Голос Джарвиса готов за {time.perf_counter() - t:.1f} с: http://{HOST}:{PORT}/tts', flush=True)
    HTTPServer((HOST, PORT), Handler).serve_forever()
    return 0


if __name__ == '__main__':
    sys.exit(main())
