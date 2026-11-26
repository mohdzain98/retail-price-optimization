# Retail Price Optimization

> 🚧 Work in progress — Phase 0 (setup) complete. See the [implementation plan](docs/implementation_plan.md).

End-to-end **Revenue Growth Management (RGM)** pipeline on the dunnhumby *Breakfast at the Frat* dataset:

1. **Price elasticity**: log-log models, from per-SKU OLS up to a **Bayesian hierarchical model (PyMC)**
2. **Baseline and decomposition**: baseline volume plus price and promo "due-to"
3. **Price optimization**: profit-maximizing prices with **NLopt**, both per SKU and across the portfolio with business constraints
4. **Forecast and promo budget**: 13-week uplift scenarios and a promo calendar under budget (MILP)
5. **Streamlit what-if simulator**

## Quickstart

```bash
make setup        # uv sync + pre-commit hooks
cp .env.example .env
make data-check   # tells you how to get the dataset if missing
make check        # lint + tests + pre-commit
```

`make setup` uses [uv](https://docs.astral.sh/uv/) if it is installed. Otherwise it creates `.venv` with
`python3 -m venv` and installs `requirements.txt` + `requirements-dev.txt` with pip (Python 3.11 or 3.12).
`requirements*.txt` list top-level packages only and mirror `pyproject.toml`. Exact versions are pinned in `uv.lock`.

## Layout

```
config/config.yaml   all tunables (margins, bounds, constraints)
src/                 pipeline package (from src.config import ...)
notebooks/           thin presentation notebooks
app/                 Streamlit app
tests/               unit tests (math + constraints)
docs/                project & implementation plans
archive/             original 2022 ANN notebook
```

## Docs

- [Project plan](docs/project_plan.md): problem framing, dataset choice, methodology
- [Implementation plan](docs/implementation_plan.md): phase-by-phase tasks
