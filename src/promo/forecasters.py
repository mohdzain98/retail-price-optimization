"""Baseline forecasting methods compared in the backtest; the winner feeds promo planning.

Each Forecaster predicts weekly baseline units for future rows from the history of
the top-down baseline: naive flat level, the lift model's seasonal index, SARIMAX
with calendar regressors per product, and a global direct multi-horizon XGBoost.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
import xgboost as xgb
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import ForecastConfig
from src.elasticity import ElasticityModel
from src.schema import HOLIDAY_COLS, Col, ForecastMethod, fourier_columns
from src.utils import AppLogger, Registry

logger = AppLogger.get_logger(__name__)

WEEK = pd.Timedelta(days=7)


class SeasonalIndex:
    """Calendar multiplier per row: exp(seasonal effect - its mean over reference rows).

    The seasonal effect is the lift model's contribution of the Fourier and holiday
    terms. Products the model cannot predict get an index of 1 (flat).
    """

    def __init__(self, model: ElasticityModel, fourier_order: int) -> None:
        self._model = model
        self._columns = [*fourier_columns(fourier_order), *HOLIDAY_COLS]

    def compute(self, target: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
        """Index aligned to target's rows; reference sets each product's level of 1."""
        effect = self._model.contribution(target, self._columns)
        ref = self._model.contribution(reference, self._columns).groupby(reference[Col.UPC]).mean()
        index = np.exp(effect - target[Col.UPC].map(ref))
        return index.fillna(1.0).rename(Col.SEASONAL_INDEX)


@dataclass(frozen=True)
class ForecastContext:
    """Shared inputs of every forecaster: settings, calendar columns and the seasonal index.

    `reference_weeks` defines the recent level (Phase 4's base volume window).
    """

    settings: ForecastConfig
    reference_weeks: int
    fourier_order: int
    index: SeasonalIndex

    @property
    def calendar(self) -> list[str]:
        """Calendar regressors known in advance: Fourier terms and holiday flags."""
        return [*fourier_columns(self.fourier_order), *HOLIDAY_COLS]

    def recent_level(self, history: pd.DataFrame) -> pd.Series:
        """Mean baseline per UPC over the last `reference_weeks` weeks of history."""
        cutoff = np.sort(history[Col.WEEK].unique())[-self.reference_weeks]
        recent = history[history[Col.WEEK] >= cutoff]
        return recent.groupby(Col.UPC)[Col.BASELINE].mean()

    def reference(self, history: pd.DataFrame) -> pd.DataFrame:
        """History rows of the last `reference_weeks` weeks."""
        cutoff = np.sort(history[Col.WEEK].unique())[-self.reference_weeks]
        return history[history[Col.WEEK] >= cutoff]


class Forecaster(ABC):
    """One way to forecast weekly baseline units for future rows of known products.

    `predict` returns a Series aligned to `future.index` (NaN where it cannot predict).
    History rows hold upc, week, baseline units, category and the calendar columns.
    """

    name: ClassVar[ForecastMethod]

    def __init__(self, ctx: ForecastContext) -> None:
        self._ctx = ctx

    @abstractmethod
    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.Series:
        """Baseline units for each future row."""


FORECASTERS: Registry[ForecastMethod, type[Forecaster]] = Registry("name")


@FORECASTERS.register
class NaiveFlat(Forecaster):
    """Recent level held flat: the mean baseline of the reference weeks (Phase 4's B)."""

    name = ForecastMethod.NAIVE

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.Series:
        """Each product's recent level."""
        return future[Col.UPC].map(self._ctx.recent_level(history)).astype(float)


@FORECASTERS.register
class SeasonalIndexForecaster(Forecaster):
    """Recent level x the Bayesian model's seasonal index (structural, no extra fitting).

    Uses the same Fourier and holiday effects that the elasticity model estimated.
    """

    name = ForecastMethod.SEASONAL

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.Series:
        """Recent level scaled by the index relative to the reference weeks."""
        level = future[Col.UPC].map(self._ctx.recent_level(history))
        return level * self._ctx.index.compute(future, self._ctx.reference(history))


@FORECASTERS.register
class SarimaxForecaster(Forecaster):
    """Per-product SARIMAX on log baseline with Fourier and holiday regressors.

    Yearly seasonality enters through the Fourier terms (dynamic harmonic regression),
    since 3 years are too short for a seasonal period of 52. A constant is added only
    without differencing. Missing weeks are skipped; failed fits use the recent level.
    """

    name = ForecastMethod.SARIMAX
    MIN_WEEKS: ClassVar[int] = 52

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.Series:
        """Exp of the SARIMAX mean forecast, product by product."""
        out = pd.Series(np.nan, index=future.index)
        level = self._ctx.recent_level(history)
        past = dict(tuple(history.sort_values(Col.WEEK).groupby(Col.UPC)))
        for upc, rows in future.sort_values(Col.WEEK).groupby(Col.UPC):
            series = past.get(upc)
            if series is None:
                continue
            out[rows.index] = self._fit_predict(series, rows, level.get(upc, np.nan))
        return out

    def _fit_predict(self, series: pd.DataFrame, rows: pd.DataFrame, level: float) -> np.ndarray:
        """Forecast one product; its recent level if history is short or the fit fails."""
        cols = [c for c in self._ctx.calendar if series[c].std() > 0]
        if len(series) < self.MIN_WEEKS:
            return np.full(len(rows), level)
        order = tuple(self._ctx.settings.sarimax_order)
        model = SARIMAX(
            np.log(series[Col.BASELINE].to_numpy()), exog=series[cols].to_numpy(float),
            order=order, trend="c" if order[1] == 0 else "n",
        )  # fmt: skip
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)  # frequency / start-parameter notes
                fit = model.fit(disp=False)
            pred = fit.forecast(len(rows), exog=rows[cols].to_numpy(float))
        except (np.linalg.LinAlgError, ValueError) as err:
            upc = int(series[Col.UPC].iloc[0])
            logger.warning("SARIMAX failed; using level", extra={"upc": upc, "error": str(err)})
            return np.full(len(rows), level)
        return np.exp(np.asarray(pred))


@FORECASTERS.register
class XgboostForecaster(Forecaster):
    """Global direct multi-horizon XGBoost on all products at once.

    Training rows pair every origin week with each step h = 1..H ahead. Target: log of
    the baseline h weeks later over the level at the origin (mean of the last N weeks).
    Features: calendar at the target week, Fourier terms at the origin, h, category.
    """

    name = ForecastMethod.XGBOOST
    ORIGIN: ClassVar[str] = "origin_"

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.Series:
        """Level at the last observed week x exp(predicted log ratio)."""
        hist = history.sort_values([Col.UPC, Col.WEEK]).reset_index(drop=True)
        hist = hist.assign(level=self._level(hist))
        categories = {c: i for i, c in enumerate(sorted(hist[Col.CATEGORY].unique()))}
        booster = self._train(hist, categories)
        origin = hist.groupby(Col.UPC).tail(1).set_index(Col.UPC)
        known = future[future[Col.UPC].isin(origin.index)]
        start = origin.loc[known[Col.UPC]]
        step = ((known[Col.WEEK].to_numpy() - start[Col.WEEK].to_numpy()) / WEEK).round()
        features = self._features(
            known[self._ctx.calendar].to_numpy(float),
            start[self._origin_cols()].to_numpy(float),
            np.clip(step, 1, self._ctx.settings.horizon_weeks),
            known[Col.CATEGORY].map(categories).fillna(-1).to_numpy(),
        )
        ratio = booster.predict(xgb.DMatrix(features))
        out = pd.Series(np.nan, index=future.index)
        out[known.index] = start["level"].to_numpy() * np.exp(ratio)
        return out

    def _level(self, hist: pd.DataFrame) -> pd.Series:
        """Rolling mean baseline over the last N observations of each product."""
        rolling = hist.groupby(Col.UPC)[Col.BASELINE].rolling(self._ctx.reference_weeks)
        return rolling.mean().reset_index(level=0, drop=True)

    def _train(self, hist: pd.DataFrame, categories: dict[str, int]) -> xgb.Booster:
        """Fit on every (origin, step) pair with a known target and a full level window."""
        by_upc = hist.groupby(Col.UPC)
        blocks, targets = [], []
        for step in range(1, self._ctx.settings.horizon_weeks + 1):
            target = by_upc[Col.BASELINE].shift(-step)
            ok = (target.notna() & hist["level"].notna()).to_numpy()
            calendar = by_upc[self._ctx.calendar].shift(-step).to_numpy(float)[ok]
            blocks.append(self._features(
                calendar, hist.loc[ok, self._origin_cols()].to_numpy(float),
                np.full(ok.sum(), step), hist.loc[ok, Col.CATEGORY].map(categories).to_numpy(),
            ))  # fmt: skip
            targets.append(np.log(target[ok] / hist.loc[ok, "level"]).to_numpy())
        params = self._ctx.settings.xgboost
        data = xgb.DMatrix(np.vstack(blocks), label=np.concatenate(targets))
        return xgb.train(
            {"objective": "reg:squarederror", "max_depth": params.max_depth,
             "eta": params.learning_rate, "subsample": params.subsample,
             "colsample_bytree": params.colsample_bytree,
             "min_child_weight": params.min_child_weight, "seed": params.seed},
            data, num_boost_round=params.n_estimators,
        )  # fmt: skip

    def _origin_cols(self) -> list[str]:
        """Fourier columns, read at the origin week (where the level was measured)."""
        return fourier_columns(self._ctx.fourier_order)

    @staticmethod
    def _features(
        calendar: np.ndarray, origin: np.ndarray, step: np.ndarray, category: np.ndarray
    ) -> np.ndarray:
        """Feature matrix: target-week calendar | origin Fourier | step | category code."""
        return np.column_stack([calendar, origin, step, category]).astype(float)
