@echo off
rem E0 one-click entry. Log: external\logs\e0.log
cd /d %~dp0..
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\e0_setup.ps1
echo.
echo Done. Log: external\logs\e0.log
pause
