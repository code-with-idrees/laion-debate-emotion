@echo off
REM Double-click to remove the logon autostart entry.
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\install_autostart.ps1" -Remove
echo.
echo Press any key to close.
pause >nul
