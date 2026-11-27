# Retail Price & Promo Optimization

> 🚧 Work in progress: Phases 0–5 done (setup, data prep, elasticity, baseline, NLopt price optimization, forecast + MILP promo budget); Phase 6 (Streamlit simulator) next.

End-to-end **Revenue Growth Management (RGM)** pipeline on the dunnhumby *Breakfast at the Frat* dataset:

1. **Price elasticity**: log-log models, from per-SKU OLS up to a **Bayesian hierarchical model (PyMC)**
2. **Baseline and decomposition**: baseline volume plus price and promo "due-to"
3. **Price optimization**: profit-maximizing prices with **NLopt**, both per SKU and across the portfolio with business constraints
4. **Forecast and promo budget**: 13-week baseline forecast (structural seasonal index vs SARIMAX vs XGBoost, picked by backtest), then a **MILP (PuLP + HiGHS)** that picks the promo tactic, depth and week per product and the budget itself (ROI frontier)
5. **Streamlit what-if simulator**

## Quickstart

```bash
make setup        # one-time: env + deps, .env, git hooks, dataset check (scripts/setup.sh)
make prep         # Phase 1: raw xlsx -> data/processed/panel.parquet
make elasticity   # Phase 2: regular + promo elasticities, 4 models, RGMx-style selection (~4 min)
make baseline     # Phase 3: RGMx top-down baseline + due-to decomposition (seconds)
make optimize     # Phase 4: NLopt profit-maximising prices + robustness (~80 s)
make promo        # Phase 5: forecast + MILP promo calendar + budget frontier (~25 s)
make check        # lint + types + tests + pre-commit
```

Data: download "Breakfast at the Frat" from [dunnhumby source files](https://www.dunnhumby.com/source-files/)
(free form) and put the `.xlsx` in `data/raw/`. `make data-check` verifies it.

`make setup` uses [uv](https://docs.astral.sh/uv/) if it is installed. Otherwise it creates `.venv` with
`python3 -m venv` (Python 3.12) and installs `requirements-dev.txt`, which includes `requirements.txt`, with pip.
`requirements*.txt` list top-level packages only and mirror `pyproject.toml`. Exact versions are pinned in `uv.lock`,
so use uv for reproducible results.

## Layout

```
config/config.yaml   all tunables (margins, bounds, constraints)
src/                 pipeline package: prep/, elasticity/, baseline/, optimization/, promo/, plots/, utils/
notebooks/           thin presentation notebooks
app/                 Streamlit app
tests/               unit tests (math + constraints)
docs/                methodology, architecture, data
archive/             original 2022 ANN notebook
```

## Docs

- [Methodology](docs/methodology.md): what each step computes and why, results, limitations and sources
- [Architecture](docs/architecture.md): code layout, data flow, extension points
- [Data](docs/data.md): getting the dataset, raw schema, processed outputs
