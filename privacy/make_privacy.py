"""Собрать launch.json, quasar.cfg и hosts без телеметрии из копий в backup/.

    py make_privacy.py      # пишет файлы в new/
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
SRC, OUT = HERE / 'backup', HERE / 'new'
OUT.mkdir(exist_ok=True)

DROP_TASKS = {'--telemetry', '--fluent-bitd'}
KEEP_TASKS = {'--mediad', '--platformd'}
HOSTS = ['clck.yandex.net', 'clck.yasmina.yango.com',
         'report.appmetrica.yandex.net', 'startup.appmetrica.yango.com']


def write(name, text):
    (OUT / name).write_bytes(text.encode('utf-8'))  # LF, без BOM


# 1. launch.json: убрать телеметрию и отправку журналов
launch = json.loads((SRC / 'launch.json').read_text(encoding='utf-8'))
tasks = launch['nodes'][0]['tasks']
kept = [t for t in tasks if not DROP_TASKS & set(t.get('arg', []))]
args = {a for t in kept for a in t.get('arg', [])}
assert KEEP_TASKS <= args, args
assert len(tasks) - len(kept) == 2, [t['arg'] for t in tasks]
launch['nodes'][0]['tasks'] = kept
write('launch.json', json.dumps(launch, indent=4, ensure_ascii=False) + '\n')
print('launch.json: оставлены', sorted(args))

# 2. quasar.cfg: адреса аналитики -> 127.0.0.1, остальной текст не трогаем
cfg = (SRC / 'quasar.cfg').read_text(encoding='utf-8')
for host in HOSTS:
    n = cfg.count('//' + host)
    assert n > 0, host
    cfg = cfg.replace('//' + host, '//127.0.0.1')
    print(f'quasar.cfg: {host} -> 127.0.0.1 ({n})')
json.loads(cfg)
write('quasar.cfg', cfg)

# 3. hosts: второй слой на случай, если адреса придут из облака
hosts = (SRC / 'hosts').read_text(encoding='utf-8').rstrip('\n') + '\n'
hosts += '# privacy: telemetry off\n' + ''.join(f'0.0.0.0 {h}\n' for h in HOSTS)
write('hosts', hosts)
print('hosts: +', len(HOSTS), 'строк')
