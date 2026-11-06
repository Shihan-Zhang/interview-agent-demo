@echo off
rem Knowledge-base MCP server: setup, tests, ingest, equivalence eval. Log: external\logs\kb_mcp.log
cd /d %~dp0..
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kb_mcp.ps1
echo.
echo Done. Log: external\logs\kb_mcp.log
pause
