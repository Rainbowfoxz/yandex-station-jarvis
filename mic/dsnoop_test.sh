#!/bin/sh
# Диагностика dsnoop на hw:0,2 при остановленной Алисе. Системный asound.conf не трогает.
export HOME=/tmp/jt
T=/tmp/jt
sz() { [ -f "$1" ] && wc -c < "$1" || echo 0; }

# pkill в busybox сравнивает 15 символов имени — quasar_launcher2 им не находится, гасим по PID.
kill -9 $(pidof quasar_launcher2) 2>/dev/null
kill -9 $(pidof maind) 2>/dev/null
sleep 2
echo "pcm2c: $(head -1 /proc/asound/card0/pcm2c/sub0/status)"

echo "T1 dsnoop RW (ждём ~4.6 МБ)"
arecord -q -D mics_dsnoop -c 16 -r 48000 -f S16_LE -d 3 -t raw $T/t1.raw 2>&1 | head -3
echo "  t1: $(sz $T/t1.raw)"

echo "T2 dsnoop MMAP"
arecord -q -M -D mics_dsnoop -c 16 -r 48000 -f S16_LE -d 3 -t raw $T/t2.raw 2>&1 | head -3
echo "  t2: $(sz $T/t2.raw)"

echo "T3 цепочка Алисы (audio_metrics > route > dsnoop), MMAP, 7 кан. (ждём ~2 МБ)"
arecord -q -M -D hw_feedback_and_mics -c 7 -r 48000 -f S16_LE -d 3 -t raw $T/t3.raw 2>&1 | head -3
echo "  t3: $(sz $T/t3.raw)"

echo "T4 цепочка Алисы + jarvis_mic 16 кГц одновременно (ждём ~2 МБ и ~96 КБ)"
arecord -q -M -D hw_feedback_and_mics -c 7 -r 48000 -f S16_LE -d 3 -t raw $T/t4a.raw 2>$T/t4a.err &
sleep 0.5
arecord -q -D jarvis_mic -c 1 -r 16000 -f S16_LE -d 2 -t raw $T/t4b.raw 2>&1 | head -3
wait
head -3 $T/t4a.err
echo "  t4a: $(sz $T/t4a.raw)  t4b: $(sz $T/t4b.raw)"
echo "jarvis_mic уровень (od первых байт):"; od -An -td2 -N32 -j 16000 $T/t4b.raw
