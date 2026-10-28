#!/bin/sh
# Установка программы «Джарвис на колонке» из распакованного архива jarvis-station-aarch64 (запускать на
# колонке от root из папки архива: sh install.sh). Нужны уже настроенные микрофон (asound.conf + sink.sh)
# и /data/opt/jarvis/pull.sh — см. README, «Колонка».
#   sh install.sh           # поставить и запустить
#   sh install.sh remove    # остановить и удалить (папку и автозапуск)
set -e
D=/data/opt/jarvis/station
HERE=$(cd "$(dirname "$0")" && pwd)

remove() {
  [ -x "$D/run.sh" ] && sh "$D/run.sh" stop || true
  mount -o remount,rw /
  rm -f /etc/init.d/S99zjarvis
  sync
  mount -o remount,ro /
  rm -rf "$D"
  echo "удалено"
}

if [ "$1" = remove ]; then remove; exit 0; fi
[ "$(id -u)" = 0 ] || { echo "нужен root"; exit 1; }
[ "$(uname -m)" = aarch64 ] || { echo "нужно 64-битное ядро (aarch64), а здесь $(uname -m)"; exit 1; }
[ -f /data/opt/jarvis/pull.sh ] || { echo "нет /data/opt/jarvis/pull.sh — сначала настройте микрофон"; exit 1; }

[ -x "$D/run.sh" ] && sh "$D/run.sh" stop || true
mkdir -p "$D" /data/opt/jarvis/og
cp -r "$HERE/jarvis" "$HERE/jarvis-kws" "$HERE/run.sh" "$HERE/lib" "$HERE/model" "$D/"
[ -f "$D/jarvis.conf" ] || cp "$HERE/jarvis.conf" "$D/"     # свои настройки не затираем
cp "$HERE/beep.wav" /data/opt/jarvis/og/beep.wav
chmod 755 "$D/jarvis" "$D/jarvis-kws" "$D/lib/ld-linux-aarch64.so.1"
chmod 600 "$D/jarvis.conf"

mount -o remount,rw /
cp "$HERE/S99zjarvis" /etc/init.d/S99zjarvis
chmod 755 /etc/init.d/S99zjarvis
sync
mount -o remount,ro /

sh "$D/run.sh" start
sleep 10
sh "$D/run.sh" status
