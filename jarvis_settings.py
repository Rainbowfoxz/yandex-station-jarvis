"""Настройки под свою колонку и ПК: переменная окружения, иначе jarvis.local.json рядом со скриптами, иначе
значение по умолчанию.

    jarvis.local.json:  {"JARVIS_HOST": "192.168.1.50", "GEMINI_ENV_FILE": "C:/keys/.env"}
"""
import json
import os
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
try:
    LOCAL = json.loads((HERE / 'jarvis.local.json').read_text(encoding='utf-8'))
except (OSError, ValueError):
    LOCAL = {}


def setting(name, default):
    return os.environ.get(name) or LOCAL.get(name) or default


HOST = setting('JARVIS_HOST', '192.168.1.100')                 # адрес колонки в домашней сети
SSH_PORT = str(setting('JARVIS_SSH_PORT', '2222'))              # dropbear на колонке
GEMINI_ENV_FILE = pathlib.Path(setting('GEMINI_ENV_FILE', HERE / '.env'))   # файл с GEMINI_API_KEY=...
