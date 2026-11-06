@echo off
rem E1-API pilot (20 episodes on SiliconFlow). Log: external\logs\e1api_pilot.log
cd /d %~dp0..
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\e1api_pilot.ps1
echo.
echo Done. Log: external\logs\e1api_pilot.log
pause
