@echo off
echo Starting HealthPulse AI...
echo.

REM Kill anything on port 8000 first
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)

REM Start FastAPI backend in a new window
start "VHA Backend (FastAPI)" cmd /k "cd /d %~dp0backend && ..\venv\Scripts\python.exe -m uvicorn app.main:app --reload"

REM Brief pause to let backend initialise
timeout /t 3 /nobreak >nul

REM Start Streamlit frontend in a new window
start "VHA Frontend (Streamlit)" cmd /k "cd /d %~dp0 && venv\Scripts\streamlit run app.py"

echo Both servers are starting.
echo   Backend  ^>  http://127.0.0.1:8000
echo   Frontend ^>  http://localhost:8501
echo.
echo Close the opened terminal windows to stop the servers.
