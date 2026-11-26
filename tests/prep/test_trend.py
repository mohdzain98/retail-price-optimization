"""Tests for src.prep.trend.LowessTrend.

Guards the leakage invariant (holdout values never change the trend) and that
the holdout carries the last training value forward.
Uses the end-to-end synthetic panel from conftest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.prep.trend import LowessTrend
from src.schema import Col, Split, driver_columns


class TestLowessTrend:
    """LowessTrend fits on train rows only and is flat over the holdout."""

    def test_ignores_holdout_values(self, panel: pd.DataFrame) -> None:
        """Shocking test-period units leaves every trend value unchanged."""
        shocked = panel.drop(columns=Col.LN_TREND)
        shocked.loc[shocked[Col.SPLIT] == Split.TEST, Col.LN_UNITS] += 5
        cfg = Config.load()
        drivers = driver_columns(cfg.features.fourier_order, cfg.features.pantry_lags)
        retrended = LowessTrend(cfg.trend.lowess_window_weeks, drivers).apply(shocked)
        key = [Col.UPC, Col.WEEK]
        np.testing.assert_allclose(
            panel.sort_values(key)[Col.LN_TREND].to_numpy(),
            retrended.sort_values(key)[Col.LN_TREND].to_numpy(),
        )

    def test_flat_over_holdout(self, panel: pd.DataFrame) -> None:
        """Each product has a single trend value across its test weeks."""
        test = panel[panel[Col.SPLIT] == Split.TEST]
        assert (test.groupby(Col.UPC)[Col.LN_TREND].nunique() == 1).all()
        assert panel[Col.LN_TREND].notna().all()

    def test_short_series_falls_back_to_mean(self) -> None:
        """Fewer than min_points train rows gives the mean residual (zero without drivers)."""
        frame = pd.DataFrame(
            {
                Col.UPC: 1,
                Col.T: range(4),
                Col.LN_UNITS: [1.0, 2.0, 3.0, 9.0],
                Col.SPLIT: [Split.TRAIN] * 3 + [Split.TEST],
            }
        )
        out = LowessTrend(78, min_points=5).apply(frame)
        np.testing.assert_allclose(out[Col.LN_TREND], 0.0, atol=1e-12)  # mean residual
