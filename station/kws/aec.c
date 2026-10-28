// Перенос aec.py. Постоянные и порядок действий — как там; комментарии о выборе значений — в aec.py.
#include "aec.h"

#include <math.h>
#include <string.h>

#define MU 0.6f
#define EPS 1e3f
#define LEAK 0.03f
#define OVER 2.0f
#define FLOOR 0.1f
#define SMOOTH 0.7f

static cpx tw[AEC_N / 2];   // поворотные множители FFT
static int rev[AEC_N];      // перестановка битов
static float window[AEC_N];  // sqrt(hanning(513))[:-1]
static int ready;

static void tables(void) {
    if (ready) return;
    int bits = 0;
    while ((1 << bits) < AEC_N) bits++;
    for (int i = 0; i < AEC_N; i++) {
        int r = 0;
        for (int b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
        rev[i] = r;
    }
    for (int k = 0; k < AEC_N / 2; k++) {
        tw[k].re = (float)cos(-2 * M_PI * k / AEC_N);
        tw[k].im = (float)sin(-2 * M_PI * k / AEC_N);
    }
    for (int n = 0; n < AEC_N; n++) window[n] = (float)sqrt(0.5 - 0.5 * cos(2 * M_PI * n / AEC_N));
    ready = 1;
}

// Комплексное БПФ на AEC_N точек на месте; inverse — без деления на N.
static void fft(cpx *a, int inverse) {
    for (int i = 0; i < AEC_N; i++)
        if (i < rev[i]) { cpx t = a[i]; a[i] = a[rev[i]]; a[rev[i]] = t; }
    for (int len = 2; len <= AEC_N; len <<= 1) {
        int step = AEC_N / len;
        for (int i = 0; i < AEC_N; i += len)
            for (int j = 0; j < len / 2; j++) {
                cpx w = tw[j * step];
                if (inverse) w.im = -w.im;
                cpx u = a[i + j], v = a[i + j + len / 2];
                cpx t = {v.re * w.re - v.im * w.im, v.re * w.im + v.im * w.re};
                a[i + j].re = u.re + t.re; a[i + j].im = u.im + t.im;
                a[i + j + len / 2].re = u.re - t.re; a[i + j + len / 2].im = u.im - t.im;
            }
    }
}

static void rfft(const float *x, cpx *spec) {   // numpy.fft.rfft на AEC_N точек
    cpx a[AEC_N];
    for (int i = 0; i < AEC_N; i++) a[i] = (cpx){x[i], 0};
    fft(a, 0);
    memcpy(spec, a, AEC_BINS * sizeof(cpx));
}

static void irfft(const cpx *spec, float *x) {  // numpy.fft.irfft: мнимые части DC и Найквиста отбрасываются
    cpx a[AEC_N];
    a[0] = (cpx){spec[0].re, 0};
    a[AEC_BLOCK] = (cpx){spec[AEC_BLOCK].re, 0};
    for (int k = 1; k < AEC_BLOCK; k++) {
        a[k] = spec[k];
        a[AEC_N - k] = (cpx){spec[k].re, -spec[k].im};
    }
    fft(a, 1);
    for (int i = 0; i < AEC_N; i++) x[i] = a[i].re / AEC_N;
}

void aec_init(Aec *a, int suppress) {
    tables();
    memset(a, 0, sizeof *a);
    for (int k = 0; k < AEC_BINS; k++) { a->power[k] = EPS; a->pe[k] = EPS; }
    a->suppress = suppress;
}

static void suppress(Aec *a, const float *e, const float *y, float *out) {
    float buf[AEC_N], frame[AEC_N];
    cpx ew[AEC_BINS], yw[AEC_BINS];
    for (int i = 0; i < AEC_BLOCK; i++) {
        buf[i] = a->prev_e[i] * window[i];
        buf[AEC_BLOCK + i] = e[i] * window[AEC_BLOCK + i];
    }
    rfft(buf, ew);
    for (int i = 0; i < AEC_BLOCK; i++) {
        buf[i] = a->prev_y[i] * window[i];
        buf[AEC_BLOCK + i] = y[i] * window[AEC_BLOCK + i];
    }
    rfft(buf, yw);
    memcpy(a->prev_e, e, sizeof a->prev_e);
    memcpy(a->prev_y, y, sizeof a->prev_y);
    for (int k = 0; k < AEC_BINS; k++) {
        a->pe[k] = SMOOTH * a->pe[k] + (1 - SMOOTH) * (ew[k].re * ew[k].re + ew[k].im * ew[k].im);
        a->py[k] = SMOOTH * a->py[k] + (1 - SMOOTH) * (yw[k].re * yw[k].re + yw[k].im * yw[k].im);
        float g = 1 - OVER * LEAK * a->py[k] / (a->pe[k] + EPS);
        g = g < FLOOR ? FLOOR : g > 1 ? 1 : g;
        ew[k].re *= g;
        ew[k].im *= g;
    }
    irfft(ew, frame);
    for (int i = 0; i < AEC_BLOCK; i++) {
        out[i] = a->tail[i] + frame[i] * window[i];
        a->tail[i] = frame[AEC_BLOCK + i] * window[AEC_BLOCK + i];
    }
}

int aec_block(Aec *a, const float *d, const float r[AEC_REFS][AEC_BLOCK], float *out) {
    float buf[AEC_N], yfull[AEC_N], y[AEC_BLOCK], e[AEC_BLOCK];
    cpx spec[AEC_REFS][AEC_BINS], acc[AEC_BINS];

    a->head = (a->head + AEC_PARTS - 1) % AEC_PARTS;  // np.roll(x, 1): свежий спектр — в часть 0
    for (int c = 0; c < AEC_REFS; c++) {
        memcpy(buf, a->prev[c], sizeof a->prev[c]);
        memcpy(buf + AEC_BLOCK, r[c], sizeof a->prev[c]);
        rfft(buf, spec[c]);
        memcpy(a->prev[c], r[c], sizeof a->prev[c]);
        memcpy(a->x[c][a->head], spec[c], sizeof spec[c]);
    }
    memset(acc, 0, sizeof acc);
    for (int c = 0; c < AEC_REFS; c++)
        for (int p = 0; p < AEC_PARTS; p++) {
            const cpx *w = a->w[c][p], *x = a->x[c][(a->head + p) % AEC_PARTS];
            for (int k = 0; k < AEC_BINS; k++) {
                acc[k].re += w[k].re * x[k].re - w[k].im * x[k].im;
                acc[k].im += w[k].re * x[k].im + w[k].im * x[k].re;
            }
        }
    irfft(acc, yfull);
    double ee = 0, dd = 0;
    int finite = 1;
    for (int i = 0; i < AEC_BLOCK; i++) {
        y[i] = yfull[AEC_BLOCK + i];
        e[i] = d[i] - y[i];
        if (!isfinite(e[i])) finite = 0;
        ee += (double)e[i] * e[i];
        dd += (double)d[i] * d[i];
    }
    if (!finite || ee > 16 * dd + 1e6) {  // фильтр сорвался — забываем его и учимся заново
        memset(a->w, 0, sizeof a->w);
        for (int i = 0; i < AEC_BLOCK; i++) { y[i] = 0; e[i] = d[i]; }
    }
    for (int k = 0; k < AEC_BINS; k++) {
        float p = 0;
        for (int c = 0; c < AEC_REFS; c++) p += spec[c][k].re * spec[c][k].re + spec[c][k].im * spec[c][k].im;
        a->power[k] = 0.9f * a->power[k] + 0.1f * p;
    }
    // СКО опоры за блок, каждый канал — относительно своего среднего (как std() в aec.py).
    double s2 = 0;
    for (int c = 0; c < AEC_REFS; c++) {
        double m = 0;
        for (int i = 0; i < AEC_BLOCK; i++) m += r[c][i];
        m /= AEC_BLOCK;
        for (int i = 0; i < AEC_BLOCK; i++) s2 += (r[c][i] - m) * (r[c][i] - m);
    }
    int active = sqrt(s2 / (AEC_REFS * AEC_BLOCK)) >= AEC_REF_ACTIVE;
    if (active) {
        cpx ef[AEC_BINS], grad[AEC_BINS];
        float g[AEC_N];
        memset(buf, 0, AEC_BLOCK * sizeof(float));
        memcpy(buf + AEC_BLOCK, e, sizeof e);
        rfft(buf, ef);
        for (int k = 0; k < AEC_BINS; k++) {
            float n = AEC_PARTS * a->power[k] + EPS;
            ef[k].re /= n;
            ef[k].im /= n;
        }
        for (int c = 0; c < AEC_REFS; c++)
            for (int p = 0; p < AEC_PARTS; p++) {
                const cpx *x = a->x[c][(a->head + p) % AEC_PARTS];
                for (int k = 0; k < AEC_BINS; k++) {  // conj(x) * ef
                    grad[k].re = x[k].re * ef[k].re + x[k].im * ef[k].im;
                    grad[k].im = x[k].re * ef[k].im - x[k].im * ef[k].re;
                }
                irfft(grad, g);
                memset(g + AEC_BLOCK, 0, AEC_BLOCK * sizeof(float));  // вторая половина отклика — нули
                rfft(g, grad);
                cpx *w = a->w[c][p];
                for (int k = 0; k < AEC_BINS; k++) {
                    w[k].re += MU * grad[k].re;
                    w[k].im += MU * grad[k].im;
                }
            }
    }
    if (a->suppress) suppress(a, e, y, out);
    else memcpy(out, e, sizeof e);
    return active;
}
