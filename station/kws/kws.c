// jarvis-kws MODEL — ловит «Джарвис» в потоке микрофона колонки (stdin: 6 каналов S16LE 48 кГц,
// как его отдаёт pull.sh) и пишет в stdout строку WAKE. Первая строка после загрузки модели — READY.
// Микрофон 0 и опоры динамика (каналы 4–5): фильтр нижних частот, прореживание до 16 кГц и вычитание
// эха (aec.c) — как MicDecoder в jarvis_wake.py. Без вычитания под музыкой имя не слышно.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aec.h"
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
#define PARTIAL_HITS 3   // под музыкой: имя в промежуточном результате столько обновлений подряд (блоков)
// После срабатывания распознавание сбрасывается и через 0,3 с снова видит хвост того же имени:
// столько кадров 48 кГц новых срабатываний не даём (2,5 с).
#define QUIET_FRAMES 120000
// «Звучит динамик» — опора выше AEC_REF_ACTIVE за последние 1,5 с: пишется в журнал у кандидата,
// а с KWS_GATE=1 имя тогда не засчитывается (так было до вычитания эха: голос Алисы ловец принимал
// за «Джарвис»).
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
    // KWS_GATE=1 — как до вычитания эха: пока звучит динамик, имя не засчитывать.
    int gate = getenv("KWS_GATE") != NULL;
    int cancel = getenv("KWS_NO_AEC") == NULL;  // KWS_NO_AEC=1 — без вычитания (сравнение на записях)
    make_taps();
    static Aec aec;
    aec_init(&aec, 1);
    puts("READY");
    fflush(stdout);

    // Каналы через фильтр и прореживание до 16 кГц: микрофон и две опоры (что играет динамик).
    static const int chans[3] = {MIC, 4, 5};
    static short in[BLOCK * CHANNELS];
    static float hist[3][TAPS - 1 + BLOCK];  // хвост прошлого блока + текущий
    static float pend[3][AEC_BLOCK + BLOCK / 3 + 1];  // 16 кГц, ещё не собрано в блок вычитания
    static short out[BLOCK / 3 + AEC_BLOCK];
    memset(hist, 0, sizeof hist);
    int phase = 0, pending = 0, hits = 0;
    size_t carry = 0;  // байт неполного кадра с прошлого чтения
    long total = 0;    // кадров 48 кГц с начала
    long quiet_until = 0;
    long played = -SPEAKER_HOLD;  // кадр, на котором динамик звучал последний раз
    for (;;) {
        size_t got = fread((char *)in + carry, 1, sizeof in - carry, stdin);
        if (got == 0) break;
        got += carry;
        size_t frames = got / (CHANNELS * 2);
        carry = got - frames * CHANNELS * 2;
        int n = 0;
        for (int c = 0; c < 3; c++) {
            float *h = hist[c];
            for (size_t i = 0; i < frames; i++) h[TAPS - 1 + i] = in[i * CHANNELS + chans[c]];
            n = 0;
            for (size_t i = phase; i < frames; i += 3) {
                float acc = 0;
                for (int k = 0; k < TAPS; k++) acc += taps[k] * h[i + k];
                pend[c][pending + n++] = acc;
            }
            memmove(h, h + frames, (TAPS - 1) * sizeof(float));
        }
        phase = (int)((phase - (long)frames) % 3 + 3) % 3;
        pending += n;
        if (carry) memmove(in, (char *)in + frames * CHANNELS * 2, carry);
        total += frames;

        // Блоками по AEC_BLOCK: микрофон минус эхо динамика (aec.c — перенос aec.py).
        int m = 0, done = 0;
        for (; pending - done >= AEC_BLOCK; done += AEC_BLOCK) {
            float r[AEC_REFS][AEC_BLOCK], clean[AEC_BLOCK];
            memcpy(r[0], pend[1] + done, sizeof r[0]);
            memcpy(r[1], pend[2] + done, sizeof r[1]);
            const float *src = pend[0] + done;
            if (aec_block(&aec, src, r, clean)) played = total;
            if (cancel) src = clean;
            for (int i = 0; i < AEC_BLOCK; i++)
                out[m++] = src[i] > 32767 ? 32767 : src[i] < -32768 ? -32768 : (short)src[i];
        }
        for (int c = 0; c < 3; c++) memmove(pend[c], pend[c] + done, (pending - done) * sizeof(float));
        pending -= done;
        if (!m) continue;
        int music = total - played < SPEAKER_HOLD;
        double seconds = 0, conf;
        if (vosk_recognizer_accept_waveform(rec, (const char *)out, m * 2)) {
            hits = 0;
            conf = name_conf(vosk_recognizer_result(rec), &seconds);
        } else if (music) {
            // Под музыкой паузы нет и окончательного результата Vosk не даёт (имя не находилось ни с
            // вычитанием, ни без) — как jarvis_wake.py, смотрим промежуточный: имя держится PARTIAL_HITS
            // обновлений подряд. Уверенности у промежуточного нет — в журнале 0.
            hits = strstr(vosk_recognizer_partial_result(rec), "джарвис") ? hits + 1 : 0;
            if (hits < PARTIAL_HITS) continue;
            hits = 0;
            conf = 1;
            vosk_recognizer_reset(rec);
        } else {
            hits = 0;
            continue;
        }
        if (conf < 0) continue;
        const char *verdict = "WAKE";
        if (conf < conf_min) verdict = "мало уверенности";
        else if (total < quiet_until) verdict = "повтор";
        else if (gate && music) verdict = "звучит динамик";
        if (seconds > 0)
            fprintf(stderr, "%.1f с: «джарвис» уверенность %.2f, %.2f с%s -> %s\n", total / 48000.0, conf,
                    seconds, music ? ", звучит динамик" : "", verdict);
        else
            fprintf(stderr, "%.1f с: «джарвис» под музыкой (промежуточный результат) -> %s\n", total / 48000.0,
                    verdict);
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
