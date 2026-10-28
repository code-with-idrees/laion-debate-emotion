@echo off
REM Live progress dashboard for the corpus download and the pipeline.
REM
REM   double-click this file, or from any folder:
REM     D:\LAION_Debate_Emotion\scripts\progress.cmd
REM
REM Everything is stored under D:\LAION_Debate_Emotion; this only cd's there and
REM runs scripts\progress.py. Ctrl-C stops the display, not the download.
cd /d "%~dp0.."
echo Working folder: %CD%
python scripts\progress.py %*
echo.
echo Stopped watching. The download keeps running in the background.
pause >nul
