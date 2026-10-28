// jarvis-kws MODEL — ловит «Джарвис» в потоке микрофона колонки (stdin: 6 каналов S16LE 48 кГц,
// как его отдаёт pull.sh) и пишет в stdout строку WAKE. Первая строка после загрузки модели — READY.
// Берётся микрофон 0, фильтр нижних частот и прореживание до 16 кГц — как MicDecoder в jarvis_wake.py,
// но без вычитания музыки.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vosk_api.h"

#define CHANNELS 6
#define MIC 0
#define TAPS 63
#define BLOCK 4800               // кадров 48 кГц за раз = 0,1 с
// Грамматика из двух слов «мигает» именем и на других словах: по промежуточным результатам (имя держится
// 3 обновления подряд) ловец 28.10 сработал на разговор в комнате. Теперь — только окончательный результат
// (после паузы ~0,5 с) и уверенность Vosk в слове не ниже CONF_MIN (меняется переменной KWS_CONF).
// Каждый кандидат пишется в stderr с уверенностью и длиной слова — по ним подбирается порог.
#define CONF_MIN 0.9
// После срабатывания распознавание сбрасывается и через 0,3 с снова видит хвост того же имени:
// столько кадров 48 кГц новых срабатываний не даём (2,5 с).
#define QUIET_FRAMES 120000
// Каналы 4–5 — опора: что сейчас играет динамик. Вычитания эха здесь нет, поэтому пока динамик звучит
// (и ещё 1,5 с после), имя не засчитываем: голос Алисы из динамика ловец принимал за «Джарвис».
// Порог — СКО опоры за блок, как REF_ACTIVE в aec.py: в тишине канал 4 — нули, 5 — смещение ~17.
#define REF_ACTIVE 20.0
#define SPEAKER_HOLD 72000

static float taps[TAPS];

static void make_taps(void) {
    // Окно Хэмминга, срез 7 кГц при 48 кГц: ниже половины новой частоты 16 кГц.
    double fc = 7000.0 / 48000.0, sum = 0;
    for (int i = 0; i < TAPS; i++) {
        int m = i - TAPS / 2;
        double h = m == 0 ? 2 * fc : sin(2 * M_PI * fc * m) / (M_PI * m);
        h *= 0.54 - 0.46 * cos(2 * M_PI * i / (TAPS - 1));
        taps[i] = (float)h;
        sum += h;
    }
    for (int i = 0; i < TAPS; i++) taps[i] /= (float)sum;
}

// СКО опорных каналов за блок (каждый канал — относительно своего среднего).
static double speaker_level(const short *in, size_t frames) {
    double sum = 0;
    for (int c = 4; c <= 5; c++) {
        double s = 0, s2 = 0;
        for (size_t i = 0; i < frames; i++) {
            double v = in[i * CHANNELS + c];
            s += v;
            s2 += v * v;
        }
        double mean = s / frames;
        sum += s2 / frames - mean * mean;
    }
    return sqrt(sum / 2);
}

// Самая уверенная «джарвис» в результате Vosk со словами: {"result":[{"conf":…,"end":…,"start":…,
// "word":"джарвис"},…],"text":…}. Возвращает уверенность (или -1) и длину слова в секундах.
static double name_conf(const char *json, double *seconds) {
    double best = -1;
    for (const char *w = strstr(json, "\"джарвис\""); w; w = strstr(w + 1, "\"джарвис\"")) {
        const char *obj = w;
        while (obj > json && *obj != '{') obj--;
        double conf = -1, start = 0, end = 0;
        const char *p;
        if ((p = strstr(obj, "\"conf\"")) && p < w) sscanf(strchr(p, ':') + 1, "%lf", &conf);
        if ((p = strstr(obj, "\"start\"")) && p < w) sscanf(strchr(p, ':') + 1, "%lf", &start);
        if ((p = strstr(obj, "\"end\"")) && p < w) sscanf(strchr(p, ':') + 1, "%lf", &end);
        if (conf > best) {
            best = conf;
            *seconds = end - start;
        }
    }
    return best;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: jarvis-kws MODEL < pcm\n");
        return 2;
    }
    vosk_set_log_level(-1);
    VoskModel *model = vosk_model_new(argv[1]);
    if (!model) {
        fprintf(stderr, "нет модели %s\n", argv[1]);
        return 1;
    }
    VoskRecognizer *rec = vosk_recognizer_new_grm(model, 16000.0, "[\"джарвис\", \"[unk]\"]");
    vosk_recognizer_set_words(rec, 1);  // уверенность и время каждого слова
    const char *env = getenv("KWS_CONF");
    double conf_min = env ? atof(env) : CONF_MIN;
    int gate = getenv("KWS_NO_GATE") == NULL;  // проверка на записях, где «Джарвис» звучит из динамика
    make_taps();
    puts("READY");
    fflush(stdout);

    static short in[BLOCK * CHANNELS];
    static float hist[TAPS - 1 + BLOCK];  // хвост прошлого блока + текущий, один канал
    static short out[BLOCK / 3 + 1];
    memset(hist, 0, sizeof hist);
    int phase = 0;
    size_t carry = 0;  // байт неполного кадра с прошлого чтения
    long total = 0;    // кадров с начала
    long quiet_until = 0;
    long played = -SPEAKER_HOLD;  // кадр, на котором динамик звучал последний раз
    for (;;) {
        size_t got = fread((char *)in + carry, 1, sizeof in - carry, stdin);
        if (got == 0) break;
        got += carry;
        size_t frames = got / (CHANNELS * 2);
        carry = got - frames * CHANNELS * 2;
        for (size_t i = 0; i < frames; i++) hist[TAPS - 1 + i] = in[i * CHANNELS + MIC];
        if (frames && speaker_level(in, frames) > REF_ACTIVE) played = total + (long)frames;
        int n = 0;
        for (size_t i = phase; i < frames; i += 3) {
            float acc = 0;
            for (int k = 0; k < TAPS; k++) acc += taps[k] * hist[i + k];
            out[n++] = acc > 32767 ? 32767 : acc < -32768 ? -32768 : (short)acc;
        }
        phase = (int)((phase - (long)frames) % 3 + 3) % 3;
        memmove(hist, hist + frames, (TAPS - 1) * sizeof(float));
        if (carry) memmove(in, (char *)in + frames * CHANNELS * 2, carry);

        total += frames;
        if (!vosk_recognizer_accept_waveform(rec, (const char *)out, n * 2)) continue;
        double seconds = 0, conf = name_conf(vosk_recognizer_result(rec), &seconds);
        if (conf < 0) continue;
        const char *verdict = "WAKE";
        if (conf < conf_min) verdict = "мало уверенности";
        else if (total < quiet_until) verdict = "повтор";
        else if (gate && total - played < SPEAKER_HOLD) verdict = "звучит динамик";
        fprintf(stderr, "%.1f с: «джарвис» уверенность %.2f, %.2f с -> %s\n", total / 48000.0, conf,
                seconds, verdict);
        if (verdict[0] == 'W') {
            quiet_until = total + QUIET_FRAMES;
            puts("WAKE");
            fflush(stdout);
        }
    }
    vosk_recognizer_free(rec);
    vosk_model_free(model);
    return 0;
}
