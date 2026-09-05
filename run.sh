#!/usr/bin/env bash
# ExamAgent launcher
set -e
cd "$(dirname "$0")"

echo "ExamAgent"

if ! command -v python >/dev/null 2>&1; then
  echo "Python 3.11+ is required and was not found on PATH." >&2
  exit 1
fi

if ! python -c "import streamlit, pydantic, sqlalchemy, sklearn, pypdf" >/dev/null 2>&1; then
  echo "Installing dependencies..."
  python -m pip install -r requirements.txt
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example (offline mode until you add an API key)."
fi

echo "Starting on http://localhost:8501 ..."
exec python -m streamlit run app.py
