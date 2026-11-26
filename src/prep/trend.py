"""Residual LOESS trend per series, used as a covariate in the elasticity model.

Mirrors RGMx: log units are first regressed on the demand drivers (price, promo,
coverage, holidays, pantry loading, season); LOESS of the residual over time is the
trend. Fitted on training weeks only and held flat into the holdout (no leakage).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from src.schema import Col, Split
from src.utils import AppLogger

logger = AppLogger.get_logger(__name__)


class LowessTrend:
    """Adds `ln_trend`: smoothed residual drift of ln_units, per store/product series.

    The window is in weeks (frac = window / n_train), so it stays smoother than
    yearly seasonality when window > 52. With no drivers it smooths raw ln_units.
    """

    SERIES_COLS: ClassVar[tuple[Col, ...]] = (Col.STORE, Col.UPC)

    def __init__(self, window_weeks: int, drivers: Sequence[str] = (), min_points: int = 5) -> None:
        self._window_weeks = window_weeks
        self._drivers = list(drivers)
        self._min_points = min_points

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return a sorted copy of panel with ln_trend added."""
        group_cols = [c for c in self.SERIES_COLS if c in panel.columns]
        ordered = panel.sort_values([*group_cols, Col.T]).reset_index(drop=True)
        parts = [self._fit_series(g) for _, g in ordered.groupby(group_cols, sort=False)]
        logger.info(
            "Added residual LOESS trend",
            extra={"series": len(parts), "drivers": len(self._drivers)},
        )
        return ordered.assign(**{Col.LN_TREND: pd.concat(parts)})

    def _fit_series(self, series: pd.DataFrame) -> pd.Series:
        """Trend for one series; test rows carry the last train value."""
        t = series[Col.T].to_numpy(float)
        is_train = (series[Col.SPLIT] == Split.TRAIN).to_numpy()
        target = self._residual(series, is_train)
        trend = np.full(len(t), target[is_train].mean() if is_train.any() else 0.0)
        if is_train.sum() >= self._min_points:
            frac = min(1.0, self._window_weeks / is_train.sum())
            fitted = lowess(target[is_train], t[is_train], frac=frac, return_sorted=False)
            trend[is_train] = fitted
            trend[~is_train] = fitted[np.argmax(t[is_train])]
        return pd.Series(trend, index=series.index)

    def _residual(self, series: pd.DataFrame, is_train: np.ndarray) -> np.ndarray:
        """ln_units minus its least-squares fit on the drivers (train rows define the fit)."""
        y = series[Col.LN_UNITS].to_numpy(float)
        train = series[is_train]
        active = [d for d in self._drivers if d in series.columns and train[d].std() > 0]
        design = np.column_stack([np.ones(len(series)), series[active].to_numpy(float)])
        coef, *_ = np.linalg.lstsq(design[is_train], y[is_train], rcond=None)
        return y - design @ coef
