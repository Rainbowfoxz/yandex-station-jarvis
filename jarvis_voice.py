"""Произвольная фраза голосом Джарвиса: клон по записям jarvis-og на F5-TTS с русской моделью ESpeech.

Запускать питоном из отдельного окружения F5-TTS (torch с CUDA 12.8 под RTX 5070):

    %USERPROFILE%\\Downloads\\f5-tts\\venv\\Scripts\\python.exe jarvis_voice.py "Текст" ["Текст 2" ...]
    ... jarvis_voice.py --ref og film story "Текст"     # сравнить образцы: out\\og, out\\film, out\\story

WAV складываются в Downloads\\f5-tts\\out, проиграть на колонке: py jarvis.py --file out\\01.wav
Образец голоса — склейка записей jarvis-og (CC BY-NC-SA 4.0, только некоммерческое использование).
Ударения расставляет ruaccent («+» перед ударной гласной); если в тексте уже есть «+», он не трогается.
"""
import argparse
import os
import pathlib
import re
import sys
import time

ROOT = pathlib.Path.home() / 'Downloads' / 'f5-tts'
# Модели — в Загрузках, а не в общем кэше на C. Без hf-xet: его загрузчик падал на нехватке памяти.
os.environ.setdefault('HF_HOME', str(ROOT / 'hf'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from ruaccent import RUAccent  # noqa: E402
from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text  # noqa: E402
from f5_tts.model import DiT  # noqa: E402

HERE = pathlib.Path(__file__).parent
CLIPS = HERE / 'jarvis-og-wav'
FILM = ROOT / 'ref'          # реплики из дубляжа фильмов, собирает voice_refs.py
# Образцы до 12 с (больше F5 обрезает): файл записи и её дословный текст, числа — словами.
REFS = {
    # Записи jarvis-og, текст которых совпал у Whisper и Gemini. Только короткие «…, сэр».
    'og': [(CLIPS / 'reply1.wav', 'Да, сэр.'), (CLIPS / 'ok3.wav', 'Запрос выполнен, сэр.'),
           (CLIPS / 'thanks.wav', 'Всегда к вашим услугам, сэр.'),
           (CLIPS / 'stupid.wav', 'Очень тонкое замечание, сэр.'), (CLIPS / 'ok2.wav', 'Загружаю, сэр.'),
           (CLIPS / 'run.wav', 'Доброе утро.')],
    # Повествовательные реплики из фильма и одна «…, сэр».
    'film': [(FILM / 'film_device.wav',
              'К сожалению, устройство, которое сохраняет вам жизнь, в то же время убивает вас.'),
             (FILM / 'film_suit.wav',
              'Судя по всему, использование костюма железного человека усугубляет ваше состояние.'),
             (CLIPS / 'stupid.wav', 'Очень тонкое замечание, сэр.')],
    # Одно длинное предложение из рассказа про Ванко.
    'story': [(FILM / 'film_vanko.wav',
               'Антон Ванко был советским физиком, бежавшим в США в тысяча девятьсот шестьдесят третьем году. '
               'Однако уже в тысяча девятьсот шестьдесят седьмом он был обвинён в шпионаже и депортирован.')],
}
REF_RATE = 24000             # частота модели
NO_COMMA_SIR = re.compile(r',\s*(?=с\+?эр\b)', re.IGNORECASE)   # «…, сэр» -> «… сэр»
# Пауза между записями в образце. F5 берёт темп речи из образца (секунды на букву): при 0,3 с
# «Джарвис» выходил медленным, и лишнее время модель заполняла паузами до 0,7 с перед «сэр».
GAP = 0.1
MAX_PAUSE = 0.2     # паузы длиннее внутри фразы ужимаем до этой длины, с
EDGE = 0.05         # тишины по краям фразы оставляем, с (было 0,8–0,95 с в начале)
PEAK = 0.8          # запас до потолка: колонка пересчитывает 24 -> 48 кГц, выбросы за 1,0 трещат
FADE = 0.01         # плавные края предложения, с
DECLICK_HZ = 5000   # треск ищем выше этой частоты
DECLICK_K = 3.0     # вспышка — пик верха больше стольких медиан окружения
MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)  # из карточки ESpeech
MODEL_REPO = 'ESpeech/ESpeech-TTS-1_SFT-256K'
# Дообученная на репликах Джарвиса (jarvis/model_2000 — 2000 шагов, только веса fp16). У 4100 (jarvis_ft_4100.pt)
# похожесть та же (0,75 против 0,74), но вспышек треска вдвое больше: 59 против 28 на 30 фразах (28.10).
FINETUNED = ROOT / 'jarvis_ft.pt'


def _load_wav(path, *_, **__):
    # torchaudio 2.11 читает звук через torchcodec, а тому нужны DLL FFmpeg — в системе их нет.
    data, rate = sf.read(str(path), dtype='float32', always_2d=True)
    return torch.from_numpy(data.T.copy()), rate


torchaudio.load = _load_wav


def build_ref(name):
    """Склейка записей образца в один WAV (24 кГц моно) -> (путь, текст)."""
    parts = []
    for path, _ in REFS[name]:
        data, rate = sf.read(str(path), dtype='float32', always_2d=True)
        mono = data.mean(axis=1)
        n = int(len(mono) * REF_RATE / rate)
        parts.append(np.interp(np.linspace(0, len(mono) - 1, n), np.arange(len(mono)), mono))
        parts.append(np.zeros(int(REF_RATE * GAP)))
    out = ROOT / f'ref_{name}.wav'
    sf.write(str(out), np.concatenate(parts).astype(np.float32), REF_RATE)
    return out, ' '.join(text for _, text in REFS[name])


def tighten(wav, rate):
    """Обрезать тишину по краям и ужать длинные паузы внутри (окна 10 мс, порог -40 дБ от пика)."""
    win = rate // 100
    n = len(wav) // win
    rms = np.sqrt(np.mean(wav[:n * win].reshape(n, win) ** 2, axis=1)) + 1e-9
    loud = 20 * np.log10(rms / rms.max()) > -40
    if not loud.any():
        return wav
    first, last = np.flatnonzero(loud)[[0, -1]]
    edge, keep = int(EDGE * 100), int(MAX_PAUSE * 100)
    windows, quiet = [], 0
    for i in range(max(first - edge, 0), min(last + edge + 1, n)):
        quiet = 0 if loud[i] else quiet + 1
        if quiet <= keep:
            windows.append(wav[i * win:(i + 1) * win])
    return np.concatenate(windows)


def declick(wav, rate):
    """Убрать треск — короткие вспышки верха (сбои вокодера): у дообученной модели их в 5 раз больше,
    чем у исходной (16 против 3 на 35 с речи, 28.10). Верх выше DECLICK_HZ по окнам 2 мс: окно, где пик
    больше DECLICK_K медиан окружения (±50 мс), приглушается до этого уровня. Низ и ровные «с», «ш» не меняются."""
    sos = butter(6, DECLICK_HZ, 'highpass', fs=rate, output='sos')
    high = sosfiltfilt(sos, wav)
    low = wav - high
    win = rate // 500
    n = len(high) // win
    if n < 3:
        return wav
    peak = np.abs(high[:n * win].reshape(n, win)).max(axis=1) + 1e-9
    limit = DECLICK_K * median_filter(peak, size=51, mode='nearest')
    gain = np.minimum(1.0, limit / peak)
    gain = np.minimum(gain, np.minimum(np.roll(gain, 1), np.roll(gain, -1)))   # и соседние окна: край вспышки
    curve = np.interp(np.arange(len(high)), np.arange(n) * win + win / 2, gain)
    return (low + high * curve).astype(np.float32)


def soften(wav, rate, lowpass=None):
    """Без треска (declick), пики не выше PEAK (колонка пересчитывает 24 -> 48 кГц) и плавные края FADE:
    tighten режет без затухания, резкий край щёлкает.
    lowpass — срез верха, Гц; по умолчанию нет: срез на 7 кГц на колонке звучал «как из бочки» (28.10)."""
    wav = declick(wav, rate)
    if lowpass:
        wav = sosfiltfilt(butter(4, lowpass, 'lowpass', fs=rate, output='sos'), wav)
    wav = wav * min(1.0, PEAK / (np.abs(wav).max() + 1e-9))
    n = min(int(FADE * rate), len(wav) // 2)
    ramp = np.linspace(0.0, 1.0, n)
    wav[:n] *= ramp
    wav[len(wav) - n:] *= ramp[::-1]
    return wav.astype(np.float32)


class Voice:
    # 16 шагов: синтез ~1,0 с вместо ~1,9 при 32, на слух разницы нет (сравнивали на колонке 28.10).
    def __init__(self, ref='film', nfe_step=16, ckpt=FINETUNED):
        """ckpt — точка дообучения на Джарвисе (по умолчанию FINETUNED, если есть); None — исходная ESpeech."""
        ckpt = ckpt if ckpt and pathlib.Path(ckpt).exists() else None
        self.nfe_step = nfe_step
        # Сначала модель: она уходит в видеопамять и освобождает обычную, памяти на ПК впритык.
        # У дообученной — основные веса, не EMA: сглаженная копия за пару тысяч шагов почти не сдвигается.
        self.model = load_model(DiT, MODEL_CFG, str(ckpt) if ckpt else hf_hub_download(MODEL_REPO, 'espeech_tts_256k.pt'),
                                vocab_file=hf_hub_download(MODEL_REPO, 'vocab.txt'), use_ema=not ckpt)
        self.vocoder = load_vocoder()
        self.accent = RUAccent()
        # Без большого словаря (−0,7 ГБ памяти): нейросеть ставит те же ударения в многосложных
        # словах, разница только в «+» на односложных (сверяли 15 фраз 28.10).
        self.accent.load(omograph_model_size='turbo3.1', use_dictionary=False, tiny_mode=False,
                         workdir=str(ROOT / 'ruaccent'))
        # transformers 5 без этого не отдаёт token_type_ids, а ONNX-модель ударений их требует.
        tokenizer = self.accent.accent_model.tokenizer
        tokenizer.model_input_names = ['input_ids', 'token_type_ids', 'attention_mask']
        self.use(ref)

    def use(self, ref):
        """Сменить образец голоса без перезагрузки модели."""
        path, text = build_ref(ref)
        self.ref_audio, self.ref_text = preprocess_ref_audio_text(str(path), self.stress(text),
                                                                  show_info=lambda *_: None)

    def stress(self, text):
        return text if '+' in text else self.accent.process_all(text)

    def synth(self, text, nfe_step=None):
        """Текст -> (звук float32, частота). nfe_step — шагов синтеза, если не как у голоса."""
        # С запятой перед «сэр» модель тянет «с-с» на ~0,2 с — на слух это пауза (Whisper слышит «с… ср»).
        text = NO_COMMA_SIR.sub(' ', text)
        wav, rate, _ = infer_process(self.ref_audio, self.ref_text, self.stress(text), self.model, self.vocoder,
                                     nfe_step=nfe_step or self.nfe_step, show_info=lambda *_: None, progress=None)
        return soften(tighten(wav, rate), rate), rate

    def say(self, text, out):
        wav, rate = self.synth(text)
        sf.write(str(out), wav, rate)
        return len(wav) / rate


def main():
    parser = argparse.ArgumentParser(description='Фраза голосом Джарвиса (F5-TTS).')
    parser.add_argument('texts', nargs='+')
    parser.add_argument('--ref', nargs='+', default=['film'], choices=sorted(REFS),
                        help='Образец голоса; несколько — для сравнения, каждый в свою папку')
    parser.add_argument('--nfe', type=int, default=16, help='Шагов синтеза: меньше — быстрее, хуже')
    parser.add_argument('--out', default=str(ROOT / 'out'))
    parser.add_argument('--base', action='store_true', help='Исходная ESpeech, без дообучения на Джарвисе')
    args = parser.parse_args()

    t = time.perf_counter()
    voice = Voice(args.ref[0], args.nfe, ckpt=None if args.base else FINETUNED)
    print(f'Модель загружена за {time.perf_counter() - t:.1f} с, GPU: {torch.cuda.is_available()}', flush=True)
    for ref in args.ref:
        voice.use(ref)
        out = pathlib.Path(args.out) / ref
        out.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(args.texts, 1):
            t = time.perf_counter()
            seconds = voice.say(text, out / f'{i:02}.wav')
            print(f'{ref}/{i:02}.wav  {seconds:.1f} с звука за {time.perf_counter() - t:.2f} с  '
                  f'«{voice.stress(text)}»', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
