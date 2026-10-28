#!/bin/sh
# Проверка отвода при остановленной Алисе: arecord читает цепочку Алисы так же, как она (MMAP, 7 кан.).
export HOME=/tmp/jt
kill -9 $(pidof quasar_launcher2) 2>/dev/null
kill -9 $(pidof maind) 2>/dev/null
sleep 2
echo "pcm2c: $(head -1 /proc/asound/card0/pcm2c/sub0/status)"
arecord -q -M -D hw_feedback_and_mics -c 7 -r 48000 -f S16_LE -d 3 -t raw /tmp/jt/alice.raw 2>/tmp/jt/arec.err &
p=$!
sleep 5
kill -9 $p 2>/dev/null
head -3 /tmp/jt/arec.err
echo "Алиса (arecord) получила: $(wc -c < /tmp/jt/alice.raw) байт (ждём ~2 016 000)"
echo "кусков в /tmp/jarvis: $(ls /tmp/jarvis/*.raw 2>/dev/null | wc -l), всего байт: $(cat /tmp/jarvis/*.raw 2>/dev/null | wc -c)"
echo "приёмник жив: $(pidof sink.sh; ps w | grep -c [s]ink.sh)"
for f in $(ls /tmp/jarvis/ | grep raw | sort -n); do cat /tmp/jarvis/$f; done > /tmp/jt/copy.raw
n=$(wc -c < /tmp/jt/copy.raw)
cmp -n $n /tmp/jt/alice.raw /tmp/jt/copy.raw && echo "копия ($n байт) совпадает с тем, что получила Алиса" || echo "копия отличается"
