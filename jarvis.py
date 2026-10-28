"""Сказать фразу голосом «в духе Джарвиса» на Станции: Piper офлайн + эффект ИИ + золотая подсветка.

    py jarvis.py "Текст"
    py jarvis.py --delay 5 --think 2500 "Текст"
    py jarvis.py --voice ruslan --no-fx "Текст"      # голоса: dmitri, denis, ruslan

Нужны: Piper в Downloads\\piper-tts\\piper\\piper.exe и модели ru_RU-*-medium.onnx рядом, ffmpeg в PATH.
"""
import argparse
import pathlib
import subprocess
import sys
import tempfile
import time
import wave

from jarvis_settings import HOST, SSH_PORT

SSH = ['ssh', '-p', SSH_PORT, '-o', 'BatchMode=yes', f'root@{HOST}']
PIPER_DIR = pathlib.Path.home() / 'Downloads' / 'piper-tts'
REMOTE_WAV = '/data/opt/jarvis/say.wav'
# Тон чуть ниже, без мутного низа, короткое «отражение» и лёгкий металл — голос ИИ, а не человека.
FX = ('asetrate=22050*0.94,aresample=22050,atempo=1/0.94,'
      'highpass=f=90,lowpass=f=9000,'
      'aecho=0.8:0.6:35|60:0.22|0.12,'
      'flanger=delay=0:depth=1:regen=10:width=40:speed=0.2')
GST = ('cd /system/vendor/quasar && LD_LIBRARY_PATH=/system/vendor/quasar:libs ./maind --gst-launch '
       f'"filesrc location={REMOTE_WAV} ! decodebin ! audioconvert ! audioresample ! alsasink"')


def synth(text, voice, fx, out):
    raw = out.with_suffix('.raw.wav')
    subprocess.run([str(PIPER_DIR / 'piper' / 'piper.exe'),
                    '--model', str(PIPER_DIR / f'ru_RU-{voice}-medium.onnx'),
                    '--output_file', str(raw)],
                   input=text.encode('utf-8'), check=True, capture_output=True)
    chain = ['-af', FX] if fx else []
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(raw), *chain,
                    '-ar', '48000', '-ac', '1', '-sample_fmt', 's16', str(out)], check=True)


def convert(src, out):
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(src),
                    '-ar', '48000', '-ac', '1', '-sample_fmt', 's16', str(out)], check=True)


def duration_ms(path):
    with wave.open(str(path)) as w:
        return int(w.getnframes() * 1000 / w.getframerate())


def main():
    parser = argparse.ArgumentParser(description='Фраза голосом в духе Джарвиса.')
    parser.add_argument('text', nargs='?', default='')
    parser.add_argument('--file', help='Проиграть готовый звуковой файл вместо синтеза')
    parser.add_argument('--voice', default='dmitri', choices=['dmitri', 'denis', 'ruslan'])
    parser.add_argument('--no-fx', action='store_true', help='Без эффекта ИИ')
    parser.add_argument('--delay', type=float, default=0, help='Пауза перед ответом, секунды')
    parser.add_argument('--think', type=int, default=0, help='Анимация «думаю» перед ответом, мс')
    args = parser.parse_args()
    if not args.file and not args.text.strip():
        parser.error('Нужен текст или --file.')

    with tempfile.TemporaryDirectory() as tmp:
        wav = pathlib.Path(tmp) / 'say.wav'
        if args.file:
            convert(args.file, wav)
        else:
            synth(args.text, args.voice, not args.no_fx, wav)
        ms = duration_ms(wav)
        # Загружаем заранее, чтобы после паузы ответ прозвучал сразу.
        with open(wav, 'rb') as f:
            subprocess.run(SSH + [f'mkdir -p /data/opt/jarvis && cat > {REMOTE_WAV}'], stdin=f, check=True)

    if args.delay > 0:
        print(f'Отвечу через {args.delay:g} с...', flush=True)
        time.sleep(args.delay)
    think = f'sh /data/opt/think/anim.sh /data/opt/think/thinking.bin {args.think}; ' if args.think > 0 else ''
    speak = f'sh /data/opt/think/anim.sh /data/opt/think/speaking.bin {ms + 300} & {GST} >/dev/null 2>&1; wait'
    subprocess.run(SSH + [think + speak], stdin=subprocess.DEVNULL, check=True)
    print('Готово.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
