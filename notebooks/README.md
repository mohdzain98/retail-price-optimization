# Notebooks

Thin, presentation-only notebooks. All logic lives in `src/`; notebooks import,
call, plot and narrate. No `def`/`class` here; keep cells under ~15 lines.

| Notebook | Step | Shows |
|---|---|---|
| `01_eda.ipynb` | data | coverage, price vs units, promo frequency, trend |
| `02_elasticity.ipynb` | elasticity | OLS vs MixedLM vs Bayesian forest plot, PPC, holdout accuracy |
| `03_baseline.ipynb` | baseline | baseline and due-to decomposition |
| `04_price_optimization.ipynb` | regular prices | NLopt solvers, recommended price changes, robustness, profit vs revenue goal |
| `05_forecast_promo.ipynb` | forecast + promotions | forecast backtest (naive, seasonal, SARIMAX, XGBoost), budget frontier, promo calendar, scenarios |

Outputs are stripped on commit (`nbstripout`, installed by `make setup`).
