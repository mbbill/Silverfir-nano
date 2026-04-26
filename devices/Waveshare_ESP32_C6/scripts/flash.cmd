@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0flash.ps1" %*
exit /b %ERRORLEVEL%
