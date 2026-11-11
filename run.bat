@echo off
title Facial Emotion Detection System
cd /d "%~dp0"
echo ========================================================
echo  AI Emotion Detection System - Starting...
echo  Modes: [1] Live Camera  [2] Upload Image
echo ========================================================
"%~dp0.venv\Scripts\python.exe" -u "%~dp0emotion_detection.py"
if %errorlevel% neq 0 (
    echo.
    echo Application exited with error code: %errorlevel%
    pause
)
