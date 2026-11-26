"""LOESS trend per series, used as a covariate in the elasticity model.

Mirrors RGMx step 02: a smooth trend of log units absorbs slow drift (distribution
gains, brand growth) so it is not attributed to price. Fitted on training weeks
only and held flat into the holdout to avoid leakage.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from src.schema import Col, Split
from src.utils import AppLogger

logger = AppLogger.get_logger(__name__)


class LowessTrend:
    """Adds `ln_trend`: LOESS of ln_units over t, per store/product series.

    The window is in weeks (frac = window / n_train), so it stays smoother than
    yearly seasonality when window > 52.
    """

    SERIES_COLS: ClassVar[tuple[Col, ...]] = (Col.STORE, Col.UPC)

    def __init__(self, window_weeks: int, min_points: int = 5) -> None:
        self._window_weeks = window_weeks
        self._min_points = min_points

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return a sorted copy of panel with ln_trend added."""
        group_cols = [c for c in self.SERIES_COLS if c in panel.columns]
        ordered = panel.sort_values([*group_cols, Col.T]).reset_index(drop=True)
        parts = [self._fit_series(g) for _, g in ordered.groupby(group_cols, sort=False)]
        logger.info("Added LOESS trend", extra={"series": len(parts)})
        return ordered.assign(**{Col.LN_TREND: pd.concat(parts)})

    def _fit_series(self, series: pd.DataFrame) -> pd.Series:
        """Trend for one series; test rows carry the last train value."""
        t = series[Col.T].to_numpy(float)
        y = series[Col.LN_UNITS].to_numpy(float)
        is_train = (series[Col.SPLIT] == Split.TRAIN).to_numpy()
        trend = np.full(len(t), y[is_train].mean() if is_train.any() else y.mean())
        if is_train.sum() >= self._min_points:
            frac = min(1.0, self._window_weeks / is_train.sum())
            fitted = lowess(y[is_train], t[is_train], frac=frac, return_sorted=False)
            trend[is_train] = fitted
            trend[~is_train] = fitted[np.argmax(t[is_train])]
        return pd.Series(trend, index=series.index)
