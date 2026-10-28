@echo off
rem Джарвис на ПК: сервер голоса (F5-TTS) и слушатель вопросов с колонки (имя ловит сама колонка).
rem Запускается при входе в Windows через start_jarvis.vbs (без окна). Журналы — в %TEMP%:
rem jarvis_tts.log и jarvis_station.log.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
start "" /b "%USERPROFILE%\Downloads\f5-tts\venv\Scripts\python.exe" -u jarvis_tts_server.py > "%TEMP%\jarvis_tts.log" 2>&1
start "" /b py -u jarvis_wake.py --station > "%TEMP%\jarvis_station.log" 2>&1
