#!/bin/sh
# Джарвис на колонке: jarvis + ловец имени, перезапуск при падении. Запускает S99zjarvis при загрузке.
#   sh run.sh start | stop | status
D=/data/opt/jarvis/station
LOG=/tmp/jarvis-station.log      # /tmp — оперативная память: журнал держим не больше ~1 МБ
PID=/tmp/jarvis-station.pid
export PATH=/bin:/sbin:/usr/bin:/usr/sbin

loop() {
  while :; do
    "$D/jarvis" -config "$D/jarvis.conf" >> "$LOG" 2>&1
    echo "$(date '+%H:%M:%S') jarvis завершился ($?), перезапуск через 5 с" >> "$LOG"
    sleep 5
  done
}

trim() {
  while :; do
    sleep 600
    if [ "$(wc -c < "$LOG")" -gt 1000000 ]; then
      tail -c 200000 "$LOG" > "$LOG.tmp" && cat "$LOG.tmp" > "$LOG"   # на месте: писатель дописывает (>>)
      rm -f "$LOG.tmp"
    fi
  done
}

stop() {
  [ -f "$PID" ] && kill $(cat "$PID") 2>/dev/null
  rm -f "$PID"
  killall jarvis 2>/dev/null
  # ловец запущен через загрузчик из lib/ — по имени процесса его не найти; pull.sh выйдет сам
  for p in $(pidof ld-linux-aarch64.so.1); do
    grep -q jarvis-kws /proc/$p/cmdline 2>/dev/null && kill $p
  done
  return 0
}

case "$1" in
  start)
    pidof jarvis >/dev/null && exit 0
    loop </dev/null >/dev/null 2>&1 &
    L=$!
    trim </dev/null >/dev/null 2>&1 &
    echo "$L $!" > "$PID"
    ;;
  stop) stop ;;
  restart) stop; sleep 1; "$0" start ;;
  status) pidof jarvis >/dev/null && echo "работает" || echo "не запущен"; tail -5 "$LOG" 2>/dev/null ;;
  *) echo "usage: $0 start|stop|restart|status"; exit 1 ;;
esac
exit 0
