@echo off
title VR Hand Gesture Control
echo ========================================
echo  VR Hand Gesture Control - Setup ^& Run
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.9+ from:
    echo https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

REM Install dependencies
echo Installing packages (first time only)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install packages. Try running:
    echo   pip install mediapipe opencv-python pyautogui numpy
    pause
    exit /b 1
)

echo.
echo Starting VR Hand Control...
echo Press Q in the overlay window to quit.
echo.
python hand_control.py

echo.
echo Press any key to exit...
pause >nul
