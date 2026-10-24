Option Explicit

Dim shell, fso, root, ps, scriptPath, command
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
ps = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
scriptPath = root & "\scripts\run_confirmatory_watchdog.ps1"
command = Chr(34) & ps & Chr(34) & " -NoProfile -ExecutionPolicy Bypass -File " & Chr(34) & scriptPath & Chr(34) & " -PollSeconds 10 -StaleSeconds 900"
shell.CurrentDirectory = root
shell.Run command, 0, False
