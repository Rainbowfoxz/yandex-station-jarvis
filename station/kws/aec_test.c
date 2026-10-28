// Сверка aec.c с aec.py: stdin — тройки int16 (микрофон, опора 1, опора 2) 16 кГц, stdout — float32
// очищенного микрофона. Сравнивает test_aec_parity.py.
#include <stdio.h>
#include "aec.h"
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

int main(void) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);   // иначе Windows правит \n и обрывает поток на 0x1A
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    static Aec a;
    aec_init(&a, 1);
    short in[AEC_BLOCK * 3];
    float d[AEC_BLOCK], r[AEC_REFS][AEC_BLOCK], out[AEC_BLOCK];
    while (fread(in, sizeof in, 1, stdin) == 1) {
        for (int i = 0; i < AEC_BLOCK; i++) {
            d[i] = in[3 * i];
            r[0][i] = in[3 * i + 1];
            r[1][i] = in[3 * i + 2];
        }
        aec_block(&a, d, r, out);
        fwrite(out, sizeof out, 1, stdout);
    }
    return 0;
}
