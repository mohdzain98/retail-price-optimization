# Notebooks

Thin, presentation-only notebooks. All logic lives in `src/`; notebooks import,
call, plot and narrate. No `def`/`class` here; keep cells under ~15 lines.

| Notebook | Phase | Shows |
|---|---|---|
| `01_eda.ipynb` | 1 | coverage, price vs units, promo frequency, trend |
| `02_elasticity.ipynb` | 2 | OLS vs MixedLM vs Bayesian forest plot, PPC, holdout accuracy |
| `03_baseline.ipynb` | 3 | baseline and due-to decomposition |
| `04_price_optimization.ipynb` | 4 | NLopt path, before/after prices, robustness |
| `05_forecast_promo.ipynb` | 5 | forecast uplift, promo calendar |

Outputs are stripped on commit (`nbstripout`, installed by `make setup`).
