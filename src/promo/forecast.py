"""Weekly baseline forecast over the planning horizon, and the method backtest.

FutureCalendar builds no-promo rows for future weeks; ForecastBacktest scores every
registered forecaster on the holdout weeks and picks the best; BaselineForecaster
runs the chosen method on the full history for the horizon.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.prep.features import FeatureStep
from src.promo.forecasters import WEEK, Forecaster
from src.schema import HOLIDAY_COLS, TACTIC_COLS, Col, ForecastMethod, Split, pantry_columns
from src.utils import AppLogger, DataValidationError

logger = AppLogger.get_logger(__name__)

AUTO = "auto"


class FutureCalendar:
    """Builds the horizon frame: one no-promo row per product and future week.

    Each row copies the product's last observed row (regular price, coverage, trend)
    and recomputes the calendar features; promo, pantry and discount terms are zero.
    """

    def __init__(
        self, horizon: int, calendar_steps: Sequence[FeatureStep], pantry_lags: int
    ) -> None:
        self._horizon = horizon
        self._steps = list(calendar_steps)
        self._promo_off = {
            **{c: 0.0 for c in (*TACTIC_COLS, *pantry_columns(pantry_lags))},
            Col.LN_PROMO_PRICE: 0.0, Col.DISCOUNT_DEPTH: 0.0, Col.PROMO_FLAG: 0,
        }  # fmt: skip

    def build(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Rows for weeks end+1 .. end+horizon with horizon_week 1..H, sorted by UPC and week."""
        last = panel.sort_values(Col.WEEK).groupby(Col.UPC).tail(1)
        steps = pd.DataFrame({Col.HORIZON_WEEK: np.arange(1, self._horizon + 1)})
        frame = last.drop(columns=[c for c in HOLIDAY_COLS if c in last]).merge(steps, how="cross")
        frame = frame.assign(
            **{
                Col.WEEK: panel[Col.WEEK].max() + frame[Col.HORIZON_WEEK] * WEEK,
                Col.T: panel[Col.T].max() + frame[Col.HORIZON_WEEK],
                Col.PRICE: frame[Col.BASE_PRICE],
                **self._promo_off,
            }
        )
        for step in self._steps:
            frame = step.apply(frame)
        return frame.sort_values([Col.UPC, Col.WEEK]).reset_index(drop=True)


class ForecastBacktest:
    """Scores every forecaster on the holdout weeks, trained on the weeks before them.

    Target is the top-down baseline (Phase 3). Reports WAPE, bias and fit time, and
    flags the method used for the horizon (configured, or lowest WAPE for `auto`).
    """

    def __init__(self, forecasters: Mapping[ForecastMethod, Forecaster], method: str) -> None:
        self._forecasters = dict(forecasters)
        self._method = method

    def run(self, baseline: pd.DataFrame) -> pd.DataFrame:
        """One row per method: WAPE, bias, scored rows, seconds and the `selected` flag."""
        test = baseline[baseline[Col.SPLIT] == Split.TEST.value]
        train = baseline[baseline[Col.SPLIT] == Split.TRAIN.value]
        columns = [Col.METHOD, Col.WAPE, Col.BIAS, Col.N_ROWS, Col.FIT_SECONDS, Col.SELECTED]
        if test.empty or train.empty:
            return pd.DataFrame(columns=columns)
        rows = []
        for method, forecaster in self._forecasters.items():
            start = time.perf_counter()
            predicted = forecaster.predict(train, test)
            rows.append(self._score(method, test[Col.BASELINE], predicted, start))
        frame = pd.DataFrame(rows)
        frame[Col.SELECTED] = frame[Col.METHOD] == self.choose(frame)
        wape = dict(zip(frame[Col.METHOD], frame[Col.WAPE].round(3), strict=True))
        logger.info("Forecast backtest", extra=wape)
        return frame[columns]

    def choose(self, scores: pd.DataFrame) -> str:
        """The configured method; for `auto`, the lowest-WAPE method (seasonal if none scored)."""
        if self._method != AUTO:
            return ForecastMethod(self._method).value
        if scores.empty:
            return ForecastMethod.SEASONAL.value
        return str(scores.loc[scores[Col.WAPE].idxmin(), Col.METHOD])

    @staticmethod
    def _score(
        method: ForecastMethod, actual: pd.Series, predicted: pd.Series, start: float
    ) -> dict:
        """WAPE and bias over the rows the method predicts, plus elapsed seconds."""
        ok = predicted.notna()
        error = predicted[ok] - actual[ok]
        total = actual[ok].sum()
        return {
            Col.METHOD: method.value,
            Col.WAPE: float(error.abs().sum() / total),
            Col.BIAS: float(error.sum() / total),
            Col.N_ROWS: int(ok.sum()),
            Col.FIT_SECONDS: time.perf_counter() - start,
        }


class BaselineForecaster:
    """Weekly baseline forecast per product at current prices with the chosen method.

    The seasonal_index column is the forecast over Phase 4's base volume B, so every
    method reads as a multiplier on the level the optimizer used.
    """

    def __init__(self, forecasters: Mapping[ForecastMethod, Forecaster]) -> None:
        self._forecasters = dict(forecasters)

    def forecast(
        self, future: pd.DataFrame, history: pd.DataFrame, base_volume: pd.Series, method: str
    ) -> pd.DataFrame:
        """UPC, week, horizon week, method, implied index and baseline at current prices.

        `history` is the full top-down baseline; products without a base volume are
        dropped, and rows the method cannot predict fall back to B.
        """
        future = future[future[Col.UPC].isin(base_volume.index)].reset_index(drop=True)
        if future.empty:
            raise DataValidationError("No product has both history and a base volume")
        level = future[Col.UPC].map(base_volume).astype(float)
        units = self._forecasters[ForecastMethod(method)].predict(history, future).fillna(level)
        out = future[[Col.UPC, Col.WEEK, Col.HORIZON_WEEK]].assign(
            **{
                Col.FORECAST_METHOD: method,
                Col.SEASONAL_INDEX: (units / level).to_numpy(),
                Col.FORECAST_CURRENT: units.to_numpy(),
            }
        )
        logger.info(
            "Forecast baseline", extra={"method": method, "products": future[Col.UPC].nunique()}
        )
        return out
