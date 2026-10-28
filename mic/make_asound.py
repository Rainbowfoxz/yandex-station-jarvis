"""Собрать asound.conf с отводом микрофонов: плагин file копирует то, что читает Алиса, в приёмник.

    py make_asound.py      # asound.conf.orig -> asound.conf

dsnoop на hw:0,2 этой модели не работает (проверено: зависает даже без Алисы), поэтому
Алиса читает микрофоны напрямую, как в прошивке, а копия идёт через pipe в /data/opt/jarvis/sink.sh.
"""
import pathlib

HERE = pathlib.Path(__file__).parent
orig = (HERE / 'asound.conf.orig').read_text(encoding='ascii')

OLD = 'pcm.capture_fault_inject {\n  type fault_inject\n  slave.pcm "hw_feedback_and_mics_raw"\n}'
NEW = 'pcm.capture_fault_inject {\n  type fault_inject\n  slave.pcm "jarvis_tap"\n}'
assert orig.count(OLD) == 1
conf = orig.replace(OLD, NEW)

conf += '''
# --- jarvis: copy of Alice mic input (7 ch, S16_LE, 48 kHz) to sink, original in /data/opt/backup-mic ---
pcm.jarvis_tap {
    type file
    slave.pcm "hw_feedback_and_mics_raw"
    file "|/data/opt/jarvis/sink.sh"
    format "raw"
}
'''
(HERE / 'asound.conf').write_bytes(conf.encode('ascii'))
print('ok', len(conf), 'bytes')
