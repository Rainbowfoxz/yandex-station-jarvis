#!/bin/sh
# think.sh [MS] — штатная анимация «Алиса думает» на LED-панели, MS миллисекунд (по умолчанию 2000).
F=/data/opt/think/thinking.bin
FRAMES=76
total=$(( ${1:-2000} / 20 ))
n=0
while [ $n -lt $total ]; do
  dd if=$F of=/dev/fb0 bs=356 skip=$(( n % FRAMES )) count=1 2>/dev/null
  usleep 10000  # ещё ~10 мс уходит на запуск dd
  n=$(( n + 1 ))
done
dd if=/dev/zero of=/dev/fb0 bs=356 count=1 2>/dev/null
