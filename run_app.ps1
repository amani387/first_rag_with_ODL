# OpenDataLoader + Gemini RAG PowerShell Launcher
Set-Location -Path $PSScriptRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Starting OpenDataLoader + Gemini RAG Server" -ForegroundColor Green
Write-Host "  UI will be available at: http://localhost:8000" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan

$uvicorn = "C:\tmp\langchain-odl-weaviate\.venv\Scripts\uvicorn.exe"

# Open browser after a brief delay
Start-Process "http://localhost:8000"

# Launch Uvicorn
& $uvicorn app.ingest_api:app --host 0.0.0.0 --port 8000 --reload
