@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0monitor.ps1" %*
exit /b %ERRORLEVEL%
