@echo off
REM Per-stage progress, runnable from any folder.
REM
REM   D:\LAION_Debate_Emotion\scripts\stages.cmd
REM   D:\LAION_Debate_Emotion\scripts\stages.cmd --only snippets
REM   D:\LAION_Debate_Emotion\scripts\stages.cmd --only empathic,voicenet,voiceclap
REM
REM Everything lives under D:\LAION_Debate_Emotion; this only cd's there first,
REM which is what python needs to find scripts\status.py.
cd /d "%~dp0.."
python scripts\status.py %*
