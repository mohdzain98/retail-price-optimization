#!/usr/bin/env bash
# One-time project setup: environment, dependencies, local files, git hooks, data check.
# Uses uv when installed, otherwise python -m venv + pip. Safe to re-run: done steps are skipped.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
USE_UV="${USE_UV:-$(command -v uv >/dev/null 2>&1 && echo 1 || echo 0)}"
MIN_MINOR=11
MAX_MINOR=12  # pymc / nlopt wheels
TOTAL=5
COUNT=0

step() {
    COUNT=$((COUNT + 1))
    printf '\n[%d/%d] %s\n' "$COUNT" "$TOTAL" "$1"
}

step "Creating the environment and installing dependencies (a few minutes the first time)"
if [ "$USE_UV" = "1" ]; then
    echo "uv found: syncing .venv from uv.lock (uv also fetches a matching Python if needed)"
    uv sync
else
    if ! command -v "$PYTHON" >/dev/null 2>&1; then
        echo "Python not found. Install Python 3.${MIN_MINOR}-3.${MAX_MINOR} or run: PYTHON=python3.12 make setup" >&2
        exit 1
    fi
    "$PYTHON" - <<PY || { echo "Python 3.${MIN_MINOR}-3.${MAX_MINOR} is required (found $("$PYTHON" --version))." >&2; exit 1; }
import sys
sys.exit(0 if (3, ${MIN_MINOR}) <= sys.version_info[:2] <= (3, ${MAX_MINOR}) else 1)
PY
    if [ -d .venv ]; then echo ".venv already exists, keeping it"; else "$PYTHON" -m venv .venv; fi
    .venv/bin/python -m pip --version >/dev/null 2>&1 || .venv/bin/python -m ensurepip --upgrade  # uv-made venvs ship without pip
    .venv/bin/python -m pip install --upgrade --progress-bar on pip
    .venv/bin/python -m pip install --progress-bar on -r requirements-dev.txt
    .venv/bin/python -m pip install --no-deps -e .
fi
echo "Dependencies installed"

step "Preparing local files"
if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
    echo "created .env from .env.example"
else
    echo ".env already in place"
fi
mkdir -p .cache data/raw data/processed reports/figures

step "Checking git"
if [ -d .git ]; then
    echo "git repository already exists"
else
    git init --initial-branch=main
fi

step "Installing pre-commit hooks (downloads the hook repositories)"
.venv/bin/pre-commit install
.venv/bin/pre-commit install-hooks

step "Checking the raw dataset"
.venv/bin/python scripts/check_data.py || echo "Dataset not ready yet; follow the steps above, then run: make data-check"

cat <<'MSG'

Setup done.

Next:
  make prep      build data/processed/panel.parquet from the raw workbook
  make check     lint, type check and tests
  make app       start the Streamlit app (Phase 6)
MSG
