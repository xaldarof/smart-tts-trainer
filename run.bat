@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run scripts\install.ps1 first:  powershell -ExecutionPolicy Bypass -File scripts\install.ps1
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m app.main %*
