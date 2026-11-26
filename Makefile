# Uses uv when installed; otherwise falls back to python -m venv + pip.
# Force the pip path with:  make setup USE_UV=0

PYTHON ?= python3
VENV   ?= .venv
USE_UV ?= $(if $(shell command -v uv 2>/dev/null),1,0)

ifeq ($(USE_UV),1)
  RUN := uv run
else
  RUN := env PATH="$(abspath $(VENV))/bin:$$PATH"
endif

.PHONY: setup hooks data-check lint format test check app

setup:  ## create env, install deps + pre-commit hooks
ifeq ($(USE_UV),1)
	uv sync
else
	@echo "uv not found -> using $(PYTHON) venv + pip (requires Python 3.11 or 3.12)"
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt -r requirements-dev.txt
	$(VENV)/bin/pip install -e . --no-deps
endif
	$(MAKE) hooks

hooks:  ## install pre-commit git hooks
	$(RUN) pre-commit install

data-check:  ## verify raw dataset is in data/raw
	$(RUN) python scripts/check_data.py

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

test:
	$(RUN) pytest

check: lint test  ## lint + tests + pre-commit hooks
	$(RUN) pre-commit run --all-files

app:
	$(RUN) streamlit run app/streamlit_app.py
