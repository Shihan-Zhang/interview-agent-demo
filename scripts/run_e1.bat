@echo off
rem E1 one-click entry. Log: external\logs\e1.log (resumable: rerun to continue)
cd /d %~dp0..
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\e1_toolmaze.ps1
echo.
echo Done. Log: external\logs\e1.log
pause
