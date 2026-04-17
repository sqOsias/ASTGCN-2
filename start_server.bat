@echo off
chcp 65001 >nul
title ASTGCN Real-time Traffic Prediction System

set BACKEND_PORT=8000
set FRONTEND_PORT=3000

echo ==================================================
echo   ASTGCN Real-time Traffic Prediction System
echo   Windows Launcher
echo ==================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+ and add to PATH.
    pause
    exit /b 1
)

:: Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Please install Node.js 16+ and add to PATH.
    pause
    exit /b 1
)

echo [1/4] Installing backend dependencies...
cd /d "%~dp0realtime_server\backend"
pip install -r requirements.txt -q

echo [2/4] Installing frontend dependencies...
cd /d "%~dp0realtime_server\frontend"
if not exist node_modules (
    call npm install
) else (
    echo      node_modules exists, skipping...
)

echo [3/4] Starting backend on port %BACKEND_PORT%...
cd /d "%~dp0realtime_server\backend"
start "ASTGCN Backend" cmd /k "python -m uvicorn main:app --host 0.0.0.0 --port %BACKEND_PORT%"

:: Wait for backend to be ready
echo      Waiting for backend to start...
timeout /t 8 /nobreak >nul

echo [4/4] Starting frontend on port %FRONTEND_PORT%...
cd /d "%~dp0realtime_server\frontend"
start "ASTGCN Frontend" cmd /k "npx vite --host 0.0.0.0 --port %FRONTEND_PORT%"

timeout /t 3 /nobreak >nul

echo.
echo ==================================================
echo   System started successfully!
echo   Frontend: http://localhost:%FRONTEND_PORT%
echo   Backend:  http://localhost:%BACKEND_PORT%
echo.
echo   Close the two new terminal windows to stop.
echo ==================================================
echo.
pause
