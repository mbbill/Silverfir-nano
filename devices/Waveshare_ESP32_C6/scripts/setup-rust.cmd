@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-rust.ps1"
exit /b %ERRORLEVEL%
