# ExamAgent launcher (Windows PowerShell)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "ExamAgent" -ForegroundColor Cyan

try { python --version | Out-Null } catch {
    Write-Host "Python 3.11+ is required and was not found on PATH." -ForegroundColor Red
    exit 1
}

# install dependencies only when something is missing
python -c "import streamlit, pydantic, sqlalchemy, sklearn, pypdf" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example (offline mode until you add an API key)." -ForegroundColor Yellow
}

Write-Host "Starting on http://localhost:8501 ..." -ForegroundColor Green
python -m streamlit run app.py
