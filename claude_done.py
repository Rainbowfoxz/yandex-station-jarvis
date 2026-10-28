"""Хук Stop для Claude Code: колонка голосом Джарвиса говорит «Запрос выполнен, сэр».

Claude Code запускает его в фоне (async) в конце каждого ответа и передаёт JSON на stdin.
Произвольный текст колонка говорит только голосом Алисы (облачный синтез), поэтому играем
готовую запись ok3 из jarvis-og — так же, как слушатель отвечает «Да, сэр», с золотой подсветкой.

Пока у сессии работают фоновые агенты (Agent, Workflow), молчим: главный чат закончил ответ, а работа
нет. Когда агент заканчивает, сессии приходит <task-notification> и она отвечает ещё раз — тогда и скажем.

    py claude_done.py                    # проверка без Claude Code
    py claude_done.py --agents журнал.jsonl   # какие агенты сессии считаются работающими
"""
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import time

from jarvis import SSH
from jarvis_listen import DURATIONS, GST

CLIP = 'ok3'        # «Запрос выполнен, сэр.»
COOLDOWN = 20.0     # несколько сессий заканчивают почти разом — говорим один раз
STAMP = pathlib.Path(tempfile.gettempdir()) / 'claude_done.stamp'
# Журнал сессии бывает больше гигабайта — дочитываем только новое, запуски без уведомления храним здесь.
STATE = pathlib.Path(tempfile.gettempdir()) / 'claude_done'
SILENT = 20 * 60    # журнал агента не менялся 20 мин — агент мёртв (сессию перезапустили, уведомления не будет)
CHUNK = 64 << 20

# SendMessage остановленному агенту продолжает его в фоне: результат с resumedAgentId, уведомление о конце
# придёт с id вызова SendMessage.
LAUNCH = re.compile(rb'"type":"tool_use","id":"(toolu_\w+)","name":"(Agent|Workflow|SendMessage)"')
RESUMED = re.compile(rb'resumedAgentId\\?":\s*\\?"(\w+)')
# Окно после id — через просмотр вперёд: несколько результатов в одной записи не съедают друг друга.
RESULT = re.compile(rb'"tool_use_id":"(toolu_\w+)"(?=([^\n]{0,800}))')
AGENT_ID = re.compile(rb'agentId: (\w+)')
# id и статус — в начале уведомления; дальше может идти весь отчёт агента, до закрывающего тега далеко.
NOTIFICATION = re.compile(rb'<task-notification>(?=([^\n]{0,1500}))')
TOOL_USE_ID = re.compile(rb'<tool-use-id>(toolu_\w+)</tool-use-id>')
TASK_ID = re.compile(rb'<task-id>(\w+)</task-id>')
STATUS = re.compile(rb'<status>(\w+)</status>')


def scan(transcript):
    """Дочитать журнал с прошлого места; вернуть {id вызова: agentId или ''} запущенных и не закончившихся."""
    state_file = STATE / (re.sub(r'\W', '_', transcript.stem) + '.json')
    try:
        state = json.loads(state_file.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        state = {}
    offset, calls, running = state.get('offset', 0), state.get('calls', {}), state.get('running', {})
    # Записи api-request-blob повторяют всю историю — давний запуск снова встречается после своего
    # уведомления. Закончившиеся вызовы помним, чтобы повтор не воскрешал их.
    finished = set(state.get('finished', []))
    if transcript.stat().st_size < offset:          # журнал начат заново
        offset, calls, running, finished = 0, {}, {}, set()
    with open(transcript, 'rb') as f:
        f.seek(offset)
        while True:
            data = f.read(CHUNK)
            end = data.rfind(b'\n') + 1
            if not end:                             # конец файла или запись ещё дописывается
                break
            data = data[:end]
            offset += end
            f.seek(offset)
            # Один проход по порядку в файле: остановка агента и его новое продолжение не должны поменяться местами.
            events = sorted([(m.start(), 0, m) for m in LAUNCH.finditer(data)]
                            + [(m.start(), 1, m) for m in RESULT.finditer(data)]
                            + [(m.start(), 2, m) for m in NOTIFICATION.finditer(data)], key=lambda e: e[0])
            for _, what, m in events:
                if what == 0:
                    if m[1].decode() not in finished:
                        calls[m[1].decode()] = m[2].decode()
                elif what == 1:
                    call, text = m[1].decode(), m[2]
                    if call not in calls or call in finished or b'"type":"tool_result"' not in text:
                        continue
                    kind = calls.pop(call)
                    if b'Async agent launched' in text:
                        agent = AGENT_ID.search(text)
                        running[call] = agent[1].decode() if agent else ''
                    elif kind == 'SendMessage':
                        agent = RESUMED.search(text)
                        if agent:
                            running[call] = agent[1].decode()
                    elif kind == 'Workflow' and b'"is_error":true' not in text and b"doesn't want to proceed" not in text:
                        running[call] = ''
                else:
                    block = m[1].split(b'</task-notification>')[0]
                    status = STATUS.search(block)
                    if status and status[1] in (b'running', b'pending'):
                        continue
                    ended = {c.decode() for c in TOOL_USE_ID.findall(block)}
                    if not ended:                   # «N background tasks stopped» — только task-id агентов
                        agents = {a.decode() for a in TASK_ID.findall(block)}
                        ended = {call for call, agent in running.items() if agent in agents}
                    finished |= ended
                    for call in ended:
                        running.pop(call, None)
                        calls.pop(call, None)       # запуск бывает записан дважды — второй ждал бы результат
    STATE.mkdir(exist_ok=True)
    state_file.write_text(json.dumps({'offset': offset, 'calls': calls, 'running': running,
                                      'finished': sorted(finished)}), encoding='utf-8')
    return running


def busy_agents(transcript):
    """Агенты, которые ещё работают: без уведомления о конце и с журналом, менявшимся последние 20 мин.
    У workflow своего журнала нет — смотрим на самый свежий журнал агентов сессии."""
    folder = transcript.with_suffix('') / 'subagents'
    now = time.time()

    def fresh(path):
        try:
            return now - path.stat().st_mtime < SILENT
        except OSError:
            return False

    return [agent or call for call, agent in scan(transcript).items()
            if (fresh(folder / f'agent-{agent}.jsonl') if agent
                else any(fresh(p) for p in folder.glob('agent-*.jsonl')))]


def main():
    if sys.argv[1:2] == ['--agents']:
        print(busy_agents(pathlib.Path(sys.argv[2])))
        return 0
    try:
        event = json.load(sys.stdin) if not sys.stdin.isatty() else {}
    except ValueError:
        event = {}
    if event.get('stop_hook_active'):
        return 0
    transcript = pathlib.Path(event.get('transcript_path') or '')
    if transcript.is_file() and busy_agents(transcript):
        return 0
    try:
        if time.time() - STAMP.stat().st_mtime < COOLDOWN:
            return 0
    except OSError:
        pass
    STAMP.touch()

    ms = DURATIONS.get(CLIP, 2000)
    subprocess.run(SSH + ['-o', 'ConnectTimeout=6',
                          f'sh /data/opt/think/anim.sh /data/opt/think/speaking.bin {ms + 300} & '
                          f'{GST.format(name=CLIP)} >/dev/null 2>&1; wait'],
                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=20, check=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
