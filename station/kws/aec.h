// Вычитание эха на колонке — перенос aec.py (station-setup) один в один: PBFDAF (NLMS в частотной
// области) по опорным каналам динамика + подавитель остаточного эха. 16 кГц, блоки по AEC_BLOCK.
#ifndef AEC_H
#define AEC_H

#define AEC_BLOCK 256       // 16 мс
#define AEC_PARTS 8         // фильтр 8 блоков = 128 мс
#define AEC_REFS 2          // опорные каналы 4 и 5
#define AEC_N (2 * AEC_BLOCK)
#define AEC_BINS (AEC_BLOCK + 1)
#define AEC_REF_ACTIVE 20.0f  // ниже этого (СКО опоры за блок) динамик молчит — фильтр не учится

typedef struct { float re, im; } cpx;

typedef struct {
    cpx w[AEC_REFS][AEC_PARTS][AEC_BINS];  // фильтр по частям
    cpx x[AEC_REFS][AEC_PARTS][AEC_BINS];  // спектры прошлых блоков опоры, x[.][head] — свежий
    int head;
    float prev[AEC_REFS][AEC_BLOCK];
    float power[AEC_BINS];
    float prev_e[AEC_BLOCK], prev_y[AEC_BLOCK], tail[AEC_BLOCK];
    float pe[AEC_BINS], py[AEC_BINS];
    int suppress;
} Aec;

void aec_init(Aec *a, int suppress);
// d — блок микрофона, r — блоки опор, out — очищенный блок (подавитель даёт задержку в один блок).
// Возвращает 1, если динамик в этом блоке звучал (опора выше AEC_REF_ACTIVE).
int aec_block(Aec *a, const float *d, const float r[AEC_REFS][AEC_BLOCK], float *out);

#endif
