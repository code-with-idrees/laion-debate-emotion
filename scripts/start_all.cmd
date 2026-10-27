@echo off
REM Double-click launcher for the long-running jobs.
REM Keeps a console window open so you can see what happened, while the actual
REM jobs run detached and keep going after the window is closed.
cd /d "%~dp0.."
echo Working folder: %CD%
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\start_all.ps1"
echo.
echo Press any key to close this window. The jobs keep running in the background.
pause >nul
