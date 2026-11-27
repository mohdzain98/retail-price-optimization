# Uses uv when installed; otherwise the .venv created by scripts/setup.sh (pip).
# Force the pip path with:  make setup USE_UV=0

VENV   ?= .venv
USE_UV ?= $(if $(shell command -v uv 2>/dev/null),1,0)

ifeq ($(USE_UV),1)
  RUN := uv run
else
  RUN := env PATH="$(abspath $(VENV))/bin:$$PATH"
endif

.PHONY: setup install hooks data-check prep elasticity baseline loc lint format test check app

setup:  ## one-time setup: env, deps, .env, git hooks, data check
	USE_UV=$(USE_UV) ./scripts/setup.sh

install:  ## dependencies only (no hooks, no .env)
ifeq ($(USE_UV),1)
	uv sync
else
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements-dev.txt
	$(VENV)/bin/pip install --no-deps -e .
endif

hooks:  ## install pre-commit git hooks
	$(RUN) pre-commit install

data-check:  ## verify raw dataset is in data/raw
	$(RUN) python scripts/check_data.py

prep:  ## Phase 1: raw workbook -> data/processed/panel.parquet
	$(RUN) python -m src.pipeline prep

elasticity:  ## Phase 2: panel -> data/processed/elasticities*.parquet
	$(RUN) python -m src.pipeline elasticity

baseline:  ## Phase 3: top-down baseline + due-to decomposition
	$(RUN) python -m src.pipeline baseline

loc:  ## lines of code: Python (gnost) + notebooks + GitHub language share
	$(RUN) python scripts/loc.py

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) mypy

format:
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

test:
	$(RUN) pytest

check: lint test  ## lint + types + tests + pre-commit hooks
	$(RUN) pre-commit run --all-files

app:
	$(RUN) streamlit run app/streamlit_app.py
