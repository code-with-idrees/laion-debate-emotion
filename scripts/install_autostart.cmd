@echo off
REM Double-click to make the download + pipeline resume automatically at logon.
REM Undo: scripts\uninstall_autostart.cmd
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\install_autostart.ps1"
echo.
echo Press any key to close.
pause >nul
