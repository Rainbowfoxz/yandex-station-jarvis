' Запускает start_jarvis.cmd без окна. Копия этого файла в папке автозагрузки
' (shell:startup) включает Джарвиса при входе в Windows; удалить копию — выключить.
Set shell = CreateObject("WScript.Shell")
here = shell.ExpandEnvironmentStrings("%USERPROFILE%") & "\Downloads\station-setup"
shell.Run "cmd /c """ & here & "\start_jarvis.cmd""", 0, False
