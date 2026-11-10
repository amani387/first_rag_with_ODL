@echo off
title OpenDataLoader + Gemini RAG Server
cd /d "%~dp0"
echo ============================================================
echo   Starting OpenDataLoader + Gemini RAG Server
echo   UI will be available at: http://localhost:8000
echo ============================================================

set VENV_UVICORN=C:\tmp\langchain-odl-weaviate\.venv\Scripts\uvicorn.exe

if not exist "%VENV_UVICORN%" (
    echo Error: Virtual environment uvicorn not found at %VENV_UVICORN%
    pause
    exit /b 1
)

start "" http://localhost:8000
"%VENV_UVICORN%" app.ingest_api:app --host 0.0.0.0 --port 8000 --reload
pause
