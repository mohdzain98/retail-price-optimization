"""Tests for src.promo.forecasters: naive, seasonal index, SARIMAX and XGBoost.

A clean synthetic history with a known yearly cycle (in the Fourier terms) checks
that SARIMAX and XGBoost learn it and beat the flat forecast; short histories and
unknown products fall back as documented. One test class per production class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.promo.forecasters import (
    ForecastContext,
    NaiveFlat,
    SarimaxForecaster,
    SeasonalIndex,
    SeasonalIndexForecaster,
    XgboostForecaster,
)
from src.schema import HOLIDAY_COLS, Col
from tests.promo.conftest import LinearLift

WEEKS, HORIZON, AMPLITUDE = 156, 13, 0.3
LEVELS = {1: 1000.0, 2: 400.0}


def _context(lift_model: LinearLift) -> ForecastContext:
    """Fourier order 1, 13 reference weeks, the shipped challenger settings."""
    return ForecastContext(Config.load().forecast, 13, 1, SeasonalIndex(lift_model, 1))


def _series(seed: int = 0) -> pd.DataFrame:
    """Two products, weekly baseline = level x exp(0.3 sin_1) x small noise."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2009-01-07", periods=WEEKS + HORIZON, freq="7D")
    angle = 2 * np.pi * weeks.day_of_year.to_numpy() / 365.25
    holidays: dict[str, int] = dict.fromkeys(HOLIDAY_COLS, 0)
    frames = [
        pd.DataFrame(
            {
                Col.UPC: upc,
                Col.WEEK: weeks,
                Col.CATEGORY: "C",
                "sin_1": np.sin(angle),
                "cos_1": np.cos(angle),
                **holidays,
                Col.BASELINE: level
                * np.exp(AMPLITUDE * np.sin(angle) + rng.normal(0, 0.02, len(weeks))),
            }
        )
        for upc, level in LEVELS.items()
    ]
    return pd.concat(frames, ignore_index=True)


def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """History = first WEEKS weeks, future = the next HORIZON weeks."""
    cutoff = np.sort(frame[Col.WEEK].unique())[WEEKS]
    return frame[frame[Col.WEEK] < cutoff], frame[frame[Col.WEEK] >= cutoff]


def _wape(future: pd.DataFrame, predicted: pd.Series) -> float:
    """Weighted absolute percentage error."""
    return float((predicted - future[Col.BASELINE]).abs().sum() / future[Col.BASELINE].sum())


class TestForecastContext:
    """ForecastContext computes the recent level."""

    def test_recent_level(self, lift_model: LinearLift) -> None:
        """Mean of the last 13 weeks per product."""
        history, _ = _split(_series())
        level = _context(lift_model).recent_level(history)
        expected = history.groupby(Col.UPC).tail(13).groupby(Col.UPC)[Col.BASELINE].mean()
        pd.testing.assert_series_equal(level, expected, check_names=False)


class TestSeasonalIndex:
    """SeasonalIndex is exp(seasonal effect - reference mean)."""

    def test_exact_index(self, lift_model: LinearLift) -> None:
        """sin_1 coef 0.3, holiday 0.2; reference mean effect 0.1 -> index exp(eff - 0.1)."""
        ref = pd.DataFrame({Col.UPC: 1, "sin_1": [0.0, 0.0], Col.HOLIDAY_WEEK: [0, 1]})
        target = pd.DataFrame({Col.UPC: [1, 2], "sin_1": [1.0, 1.0], Col.HOLIDAY_WEEK: 0})
        index = SeasonalIndex(lift_model, fourier_order=1).compute(target, ref)
        assert index.iloc[0] == pytest.approx(np.exp(0.3 - 0.1))
        assert index.iloc[1] == 1.0  # no reference rows for product 2


class TestNaiveFlat:
    """NaiveFlat holds the recent level."""

    def test_flat(self, lift_model: LinearLift) -> None:
        """Constant per product; unknown products are NaN."""
        history, future = _split(_series())
        future = pd.concat([future, future.head(1).assign(**{Col.UPC: 99})])
        pred = NaiveFlat(_context(lift_model)).predict(history, future)
        assert pred.groupby(future[Col.UPC]).nunique().loc[[1, 2]].eq(1).all()
        assert pred.isna().sum() == 1


class TestSeasonalIndexForecaster:
    """SeasonalIndexForecaster = level x the lift model's index."""

    def test_matches_index(self, lift_model: LinearLift) -> None:
        """Ratio to the naive level equals the seasonal index."""
        history, future = _split(_series())
        ctx = _context(lift_model)
        ratio = SeasonalIndexForecaster(ctx).predict(history, future) / NaiveFlat(ctx).predict(
            history, future
        )
        np.testing.assert_allclose(ratio, ctx.index.compute(future, ctx.reference(history)))


class TestSarimaxForecaster:
    """SarimaxForecaster learns the cycle through the Fourier regressors."""

    def test_beats_flat(self, lift_model: LinearLift) -> None:
        """Accurate on a clean cycle and better than the flat level."""
        history, future = _split(_series())
        ctx = _context(lift_model)
        sarimax = _wape(future, SarimaxForecaster(ctx).predict(history, future))
        assert sarimax < 0.05
        assert sarimax < _wape(future, NaiveFlat(ctx).predict(history, future))

    def test_short_history_uses_level(self, lift_model: LinearLift) -> None:
        """Fewer than 52 weeks: the recent level."""
        history, future = _split(_series())
        short = history.groupby(Col.UPC).tail(30)
        ctx = _context(lift_model)
        np.testing.assert_allclose(
            SarimaxForecaster(ctx).predict(short, future), NaiveFlat(ctx).predict(short, future)
        )


class TestXgboostForecaster:
    """XgboostForecaster learns the cycle across products."""

    def test_beats_flat(self, lift_model: LinearLift) -> None:
        """Accurate on a clean cycle and better than the flat level."""
        history, future = _split(_series())
        ctx = _context(lift_model)
        xgboost = _wape(future, XgboostForecaster(ctx).predict(history, future))
        assert xgboost < 0.06
        assert xgboost < _wape(future, NaiveFlat(ctx).predict(history, future))

    def test_unknown_product(self, lift_model: LinearLift) -> None:
        """A product with no history is not predicted."""
        history, future = _split(_series())
        pred = XgboostForecaster(_context(lift_model)).predict(
            history[history[Col.UPC] == 1], future
        )
        assert pred[future[Col.UPC] == 2].isna().all()
        assert pred[future[Col.UPC] == 1].notna().all()
        assert pred[future[Col.UPC] == 1].mean() == pytest.approx(
            future.loc[future[Col.UPC] == 1, Col.BASELINE].mean(), rel=0.1
        )
