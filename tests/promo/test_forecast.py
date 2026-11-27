"""Tests for src.promo.forecast: future calendar, method backtest and baseline forecast.

Use the synthetic panel from the shared conftest and the linear stub lift model,
whose seasonal coefficients make the index exactly computable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.prep.features import FourierSeasonality, HolidayFlags
from src.promo.forecast import BaselineForecaster, ForecastBacktest, FutureCalendar
from src.promo.forecasters import FORECASTERS, ForecastContext, Forecaster, SeasonalIndex
from src.schema import TACTIC_COLS, Col, ForecastMethod
from tests.promo.conftest import LinearLift

HORIZON = 6


def _calendar(cfg: Config) -> FutureCalendar:
    """Calendar with the configured Fourier and holiday steps."""
    steps = [FourierSeasonality.from_config(cfg), HolidayFlags.from_config(cfg)]
    return FutureCalendar(HORIZON, steps, cfg.features.pantry_lags)


class TestFutureCalendar:
    """FutureCalendar continues each product weekly with promotions off."""

    def test_weeks_and_promo_off(self, cfg: Config, panel: pd.DataFrame) -> None:
        """H rows per product, consecutive weeks after the last, no promo terms."""
        future = _calendar(cfg).build(panel)
        assert len(future) == panel[Col.UPC].nunique() * HORIZON
        first = future[future[Col.UPC] == future[Col.UPC].iloc[0]]
        expected = panel[Col.WEEK].max() + pd.Timedelta(weeks=1) * np.arange(1, HORIZON + 1)
        assert first[Col.WEEK].tolist() == list(expected)
        assert (future[list(TACTIC_COLS)] == 0).all().all()
        assert (future[Col.PROMO_FLAG] == 0).all() and (future[Col.LN_PROMO_PRICE] == 0).all()
        assert (future[Col.PRICE] == future[Col.BASE_PRICE]).all()

    def test_calendar_features_recomputed(self, cfg: Config, panel: pd.DataFrame) -> None:
        """Fourier terms match the future dates; holiday columns are not duplicated."""
        future = _calendar(cfg).build(panel)
        angle = 2 * np.pi * future[Col.WEEK].dt.day_of_year / 365.25
        np.testing.assert_allclose(future["sin_1"], np.sin(angle))
        assert Col.HOLIDAY_WEEK in future and f"{Col.HOLIDAY_WEEK}_x" not in future


def _forecasters(cfg: Config, lift_model: LinearLift) -> dict[ForecastMethod, Forecaster]:
    """Every registered forecaster with the shipped settings."""
    ctx = ForecastContext(cfg.forecast, 13, cfg.features.fourier_order,
                          SeasonalIndex(lift_model, cfg.features.fourier_order))  # fmt: skip
    return {cls.name: cls(ctx) for cls in FORECASTERS}


class TestBaselineForecaster:
    """BaselineForecaster runs the chosen method; missing rows fall back to B."""

    def test_level(self, cfg: Config, panel: pd.DataFrame, lift_model: LinearLift) -> None:
        """Seasonal method: forecast / B is the index; products without B are dropped."""
        upcs = panel[Col.UPC].unique()[:2]
        base = pd.Series([100.0, 50.0], index=upcs)
        history = panel.assign(**{Col.BASELINE: panel[Col.UNITS] * 0.8})
        forecaster = BaselineForecaster(_forecasters(cfg, lift_model))
        out = forecaster.forecast(_calendar(cfg).build(panel), history, base, "seasonal_index")
        assert set(out[Col.UPC]) == set(upcs)
        assert (out[Col.FORECAST_METHOD] == "seasonal_index").all()
        ratio = out[Col.FORECAST_CURRENT] / out[Col.UPC].map(base)
        np.testing.assert_allclose(ratio, out[Col.SEASONAL_INDEX])


class TestForecastBacktest:
    """ForecastBacktest scores every method and picks one."""

    def test_methods(self, cfg: Config, panel: pd.DataFrame, lift_model: LinearLift) -> None:
        """One row per method, finite WAPE, auto selects the lowest WAPE."""
        baseline = panel.assign(**{Col.BASELINE: panel[Col.UNITS] * 0.8})
        backtest = ForecastBacktest(_forecasters(cfg, lift_model), "auto")
        result = backtest.run(baseline)
        assert set(result[Col.METHOD]) == {m.value for m in ForecastMethod}
        assert np.isfinite(result[Col.WAPE]).all()
        best = result.loc[result[Col.WAPE].idxmin(), Col.METHOD]
        assert result.loc[result[Col.SELECTED], Col.METHOD].tolist() == [best]

    def test_configured_method_wins(self, cfg: Config, lift_model: LinearLift) -> None:
        """A configured method is used whatever the scores."""
        backtest = ForecastBacktest(_forecasters(cfg, lift_model), "naive_flat")
        scores = pd.DataFrame({Col.METHOD: ["naive_flat", "xgboost"], Col.WAPE: [0.3, 0.1]})
        assert backtest.choose(scores) == "naive_flat"
