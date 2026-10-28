#!/bin/sh
# Приёмник копии микрофонов от ALSA-плагина file (Алиса пишет сюда через pipe).
# Никогда не блокирует: режет поток на куски в /tmp (оперативная память), хранит последние KEEP.
# Куски: /tmp/jarvis/<номер>.raw — 6 каналов, S16_LE, 48 кГц (как читает Алиса).
# Кусок = 172032 байта (14336 кадров по 12 байт, 0,3 с): fullblock, чтобы не обрывался посреди кадра.
D=/tmp/jarvis
KEEP=40
mkdir -p $D
rm -f $D/*.raw $D/c.tmp
n=0
while :; do
  dd bs=14336 count=12 iflag=fullblock of=$D/c.tmp 2>/dev/null
  [ -s $D/c.tmp ] || exit 0          # писатель закрыл канал — выходим
  mv $D/c.tmp $D/$n.raw
  rm -f $D/$((n - KEEP)).raw
  n=$((n + 1))
done
