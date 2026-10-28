#!/bin/sh
# Сборка архива jarvis-station-aarch64.tar.gz на ПК (Git Bash / Linux). Нужны:
#   ZIG   — zig 0.13 (кросс-компилятор C):        https://ziglang.org/download/
#   GO    — go 1.22+:                             https://go.dev/dl/
#   VOSK  — распакованный vosk-linux-aarch64-0.3.45 (libvosk.so, vosk_api.h) из релизов alphacep/vosk-api
#   LIBS  — aarch64-библиотеки Debian 11 (bullseye): ld-linux-aarch64.so.1, libc.so.6, libm.so.6,
#           libpthread.so.0, libdl.so.2, libgcc_s.so.1, libstdc++.so.6, libatomic.so.1
#           (пакеты libc6, libstdc++6, libgcc-s1, libatomic1 для arm64)
#   MODEL — vosk-model-small-ru-0.22
# Пример: ZIG=~/zig/zig GO=~/go/bin/go VOSK=~/vosk-aarch64 LIBS=~/debian-arm64 MODEL=~/vosk-model-small-ru-0.22 sh build.sh
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/dist/jarvis-station-aarch64
rm -rf "$OUT" && mkdir -p "$OUT/lib"

# Ловец имени: C + libvosk под aarch64 с glibc 2.31 из lib/.
(cd "$VOSK" && "$ZIG" cc -target aarch64-linux-gnu.2.31 -mcpu=cortex_a53 -O2 -s -Wall -I. \
  "$HERE/kws/kws.c" libvosk.so -lm -o "$OUT/jarvis-kws")
# Основная программа: Go без cgo — один статический файл.
(cd "$HERE/jarvis" && GOOS=linux GOARCH=arm64 CGO_ENABLED=0 "$GO" build -trimpath -ldflags="-s -w" -o "$OUT/jarvis" .)

cp "$VOSK/libvosk.so" "$OUT/lib/"
for f in ld-linux-aarch64.so.1 libc.so.6 libm.so.6 libpthread.so.0 libdl.so.2 libgcc_s.so.1 libstdc++.so.6 libatomic.so.1; do
  cp "$LIBS/$f" "$OUT/lib/"
done
mkdir -p "$OUT/model"
cp -r "$MODEL/am" "$MODEL/conf" "$MODEL/graph" "$MODEL/ivector" "$OUT/model/"
cp "$HERE/run.sh" "$HERE/S99zjarvis" "$HERE/jarvis.conf" "$HERE/install.sh" "$HERE/beep.wav" "$HERE/NOTICE.txt" "$OUT/"
(cd "$HERE/dist" && tar czf jarvis-station-aarch64.tar.gz jarvis-station-aarch64)
ls -la "$HERE/dist/jarvis-station-aarch64.tar.gz"
