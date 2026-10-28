#!/bin/sh
# anim.sh FILE [MS] — кадры FILE (по 356 байт на кадр) на LED-панели MS миллисекунд (по умолчанию 2000).
F=$1
FRAMES=$(( $(wc -c < "$F") / 356 ))
total=$(( ${2:-2000} / 20 ))
n=0
while [ $n -lt $total ]; do
  dd if="$F" of=/dev/fb0 bs=356 skip=$(( n % FRAMES )) count=1 2>/dev/null
  usleep 10000  # ещё ~10 мс уходит на запуск dd
  n=$(( n + 1 ))
done
dd if=/dev/zero of=/dev/fb0 bs=356 count=1 2>/dev/null
