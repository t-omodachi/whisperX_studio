@echo off
title WhisperX Studio - Local Video Transcriber
cd /d "%~dp0"
echo Starting WhisperX Studio on http://127.0.0.1:8765 ...
start "" "http://127.0.0.1:8765"
".venv\Scripts\python.exe" app.py
pause

