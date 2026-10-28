"""Дообучение F5 (ESpeech) на репликах Джарвиса: штатный f5_tts.train.finetune_cli с поправками под Windows.

  * загрузчиков данных 2, а не 16: в Windows каждый процесс заново грузит torch (+2 ГБ резерва памяти);
  * torchaudio.load -> soundfile (torchcodec без DLL FFmpeg не работает) — на уровне модуля,
    чтобы подмена была и в процессах-загрузчиках (spawn импортирует этот файл заново).

Данные — Downloads\\f5-tts\\venv\\Lib\\data\\jarvis_custom (dataset_build.py export + prepare_csv_wavs),
контрольные точки — ...\\venv\\Lib\\ckpts\\jarvis. Запускать питоном из окружения F5-TTS:

    ... jarvis_finetune.py [аргументы finetune_cli поверх DEFAULTS]
    ... jarvis_finetune.py --from jarvis/model_2000.pt --run jarvis2   # продолжить с точки, новый прогон

--from: веса точки (основные, не EMA) кладутся в формат исходной ESpeech — без оптимизатора и номера шага,
иначе обучение подхватит старое расписание скорости, уже спущенное почти до нуля. --run: имя прогона
(папки ckpts\\<run> и data\\<run>_custom — копия данных jarvis_custom).
"""
import os
import pathlib
import shutil
import sys

ROOT = pathlib.Path.home() / 'Downloads' / 'f5-tts'
LIB = ROOT / 'venv' / 'Lib'
os.environ.setdefault('HF_HOME', str(ROOT / 'hf'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('ACCELERATE_MIXED_PRECISION', 'bf16')   # RTX 5070: вдвое меньше памяти на активации

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

WORKERS = 2


def _load_wav(path, *_, **__):
    data, rate = sf.read(str(path), dtype='float32', always_2d=True)
    return torch.from_numpy(data.T.copy()), rate


torchaudio.load = _load_wav


def main():
    from huggingface_hub import hf_hub_download
    from f5_tts.model import trainer
    from f5_tts.train import finetune_cli

    train = trainer.Trainer.train

    def few_workers(self, dataset, num_workers=WORKERS, resumable_with_seed=None):
        return train(self, dataset, num_workers=WORKERS, resumable_with_seed=resumable_with_seed)
    trainer.Trainer.train = few_workers

    repo = 'ESpeech/ESpeech-TTS-1_SFT-256K'
    base = hf_hub_download(repo, 'espeech_tts_256k.pt')
    args, run, start = sys.argv[1:], 'jarvis', base
    if '--from' in args:
        i = args.index('--from')
        source, run = LIB / 'ckpts' / args[i + 1], args[args.index('--run') + 1]
        args = [a for n, a in enumerate(args) if n not in (i, i + 1, args.index('--run'), args.index('--run') + 1)]
        start = continue_from(source, base, run)
    # ~210 с звука = ~7 пакетов по 3200 кадров на эпоху: 300 эпох ≈ 2100 шагов, точка каждые 250.
    defaults = ['--exp_name', 'F5TTS_v1_Base', '--dataset_name', run, '--finetune',
                '--pretrain', str(start),
                '--tokenizer', 'custom', '--tokenizer_path', hf_hub_download(repo, 'vocab.txt'),
                '--learning_rate', '1e-5', '--batch_size_per_gpu', '3200', '--batch_size_type', 'frame',
                '--epochs', '300', '--num_warmup_updates', '100',
                '--save_per_updates', '250', '--last_per_updates', '250', '--keep_last_n_checkpoints', '-1',
                '--bnb_optimizer']
    sys.argv = [sys.argv[0], *defaults, *args]
    finetune_cli.main()


def export(source, out):
    """Точка -> файл для синтеза (jarvis_voice.FINETUNED): только основные веса в fp16."""
    weights = torch.load(source, map_location='cpu', weights_only=True)['model_state_dict']
    torch.save({'model_state_dict': {k: v.half() if v.is_floating_point() else v for k, v in weights.items()}}, out)


def continue_from(source, base, run):
    """Точка дообучения -> стартовый файл в формате ESpeech (только EMA-словарь с весами точки)."""
    out = LIB / 'ckpts' / f'start_{run}.pt'
    if not out.exists():
        trained = torch.load(source, map_location='cpu', weights_only=True)['model_state_dict']
        template = torch.load(base, map_location='cpu', weights_only=True)['ema_model_state_dict']
        ema = {k: (trained[k[len('ema_model.'):]] if k.startswith('ema_model.') else v) for k, v in template.items()}
        torch.save({'ema_model_state_dict': ema}, out)
    data = LIB / 'data'
    if not (data / f'{run}_custom').exists():
        shutil.copytree(data / 'jarvis_custom', data / f'{run}_custom')
    return out


if __name__ == '__main__':
    main()
