"""Вычитание эха: убираем из микрофона то, что играет сам динамик (опорные каналы колонки).

Два каскада на 16 кГц:
1. Блочный адаптивный фильтр в частотной области (PBFDAF, нормированный LMS): фильтр длиной
   PARTS*BLOCK отсчётов (128 мс) на каждый опорный канал вычитает линейную часть эха (~16 дБ).
2. Подавитель остаточного эха: там, где по оценке фильтра ещё звучит музыка, частоты приглушаются
   (усиление Винера с полом FLOOR), частоты с голосом остаются. Окна sqrt-Ханна с перекрытием 50 %.
"""
import numpy as np

BLOCK = 256        # 16 мс при 16 кГц
PARTS = 8          # 8 блоков = 128 мс: на записи с музыкой лучше 16 и 32 частей (16,5 против 15 и 12 дБ)
MU = 0.6           # шаг адаптации
EPS = 1e3          # регуляризация нормировки, чтобы не делить на тишину
REF_ACTIVE = 20.0  # ниже этого (СКО опоры за блок) динамик молчит: в тишине канал 4 — нули, 5 — смещение ~17
LEAK = 0.03        # доля мощности эха после первого каскада (16,5 дБ подавления на записи с музыкой)
OVER = 2.0         # с каким запасом считать остаток эха
FLOOR = 0.1        # не глушить частоту сильнее, чем на 20 дБ
SMOOTH = 0.7       # сглаживание спектров подавителя между блоками
WINDOW = np.sqrt(np.hanning(2 * BLOCK + 1)[:-1])


class EchoCanceller:
    def __init__(self, refs, suppress=True):
        self.refs = refs
        self.suppress = suppress
        bins = BLOCK + 1
        self.w = np.zeros((refs, PARTS, bins), complex)      # фильтр по частям
        self.x = np.zeros((refs, PARTS, bins), complex)      # спектры прошлых блоков опоры
        self.prev = np.zeros((refs, BLOCK))
        self.power = np.full(bins, EPS)
        self.mic_rest = np.zeros(0)
        self.ref_rest = np.zeros((refs, 0))
        # Подавитель: прошлые блоки ошибки и оценки эха, сглаженные спектры, хвост перекрытия.
        self.prev_e = np.zeros(BLOCK)
        self.prev_y = np.zeros(BLOCK)
        self.pe = np.full(bins, EPS)
        self.py = np.zeros(bins)
        self.tail = np.zeros(BLOCK)

    def process(self, mic, ref):
        """mic: (n,), ref: (refs, n) float, 16 кГц. Возвращает очищенный сигнал с задержкой
        до двух блоков (хвост, не кратный блоку, отдаётся при следующем вызове)."""
        mic = np.concatenate([self.mic_rest, mic])
        ref = np.concatenate([self.ref_rest, ref], axis=1)
        whole = len(mic) // BLOCK * BLOCK
        out = np.empty(whole)
        for start in range(0, whole, BLOCK):
            out[start:start + BLOCK] = self._block(mic[start:start + BLOCK], ref[:, start:start + BLOCK])
        self.mic_rest, self.ref_rest = mic[whole:], ref[:, whole:]
        return out

    def _block(self, d, r):
        spec = np.fft.rfft(np.concatenate([self.prev, r], axis=1), axis=1)   # (refs, bins)
        self.prev = r
        self.x = np.roll(self.x, 1, axis=1)
        self.x[:, 0] = spec
        y = np.fft.irfft((self.w * self.x).sum(axis=(0, 1)))[BLOCK:]
        e = d - y
        if not np.isfinite(e).all() or (e ** 2).sum() > 16 * (d ** 2).sum() + 1e6:
            # Страховка: фильтр сорвался — забываем его и учимся заново, а не глохнем навсегда.
            self.w[:] = 0
            y, e = np.zeros(BLOCK), d.copy()
        self.power = 0.9 * self.power + 0.1 * (np.abs(spec) ** 2).sum(axis=0)
        if (r - r.mean(axis=1, keepdims=True)).std() < REF_ACTIVE:
            # Динамик молчит: учиться не на чем. Без этой остановки фильтр дрейфует на шуме
            # и речи в комнате, а на первом же звуке динамика выдаёт огромную ошибку.
            return self._suppress(e, y) if self.suppress else e
        ef = np.fft.rfft(np.concatenate([np.zeros(BLOCK), e]))
        # Нормировка на мощность всех PARTS частей опоры, иначе шаг в PARTS раз больше и фильтр разносит.
        grad = np.conj(self.x) * (ef / (PARTS * self.power + EPS))
        # Ограничение градиента: вторая половина импульсной характеристики должна быть нулевой.
        g = np.fft.irfft(grad, axis=2)
        g[..., BLOCK:] = 0
        self.w += MU * np.fft.rfft(g, axis=2)
        return self._suppress(e, y) if self.suppress else e

    def _suppress(self, e, y):
        ew = np.fft.rfft(np.concatenate([self.prev_e, e]) * WINDOW)
        yw = np.fft.rfft(np.concatenate([self.prev_y, y]) * WINDOW)
        self.prev_e, self.prev_y = e, y
        self.pe = SMOOTH * self.pe + (1 - SMOOTH) * np.abs(ew) ** 2
        self.py = SMOOTH * self.py + (1 - SMOOTH) * np.abs(yw) ** 2
        # Остаток эха — доля LEAK от оценки эха (то, что линейный фильтр не взял): Винер по нему.
        gain = np.clip(1 - OVER * LEAK * self.py / (self.pe + EPS), FLOOR, 1)
        frame = np.fft.irfft(ew * gain) * WINDOW
        out = self.tail + frame[:BLOCK]
        self.tail = frame[BLOCK:]
        return out
