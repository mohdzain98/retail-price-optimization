"""Tests for src.prep.features: log features, seasonality, holidays, split, builder.

Holiday checks use known reference dates (Thanksgiving 2009, Super Bowl 2010).
The split test guards the key leakage invariant: every train week precedes
every test week. One test class per production class.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.prep.features import (
    FEATURE_STEPS,
    FeatureBuilder,
    FourierSeasonality,
    HolidayFlags,
    LogFeatures,
    TimeSplit,
)
from src.schema import Col, Split
from src.utils import ConfigError


def _weeks(dates: list[str]) -> pd.DataFrame:
    """Minimal panel with only a week column."""
    return pd.DataFrame({Col.WEEK: pd.to_datetime(dates)})


class TestLogFeatures:
    """LogFeatures adds logs and a clipped discount depth."""

    def test_values(self) -> None:
        """Discount depth is 1 - price/base; logs match numpy."""
        frame = pd.DataFrame(
            {Col.UNITS: [10, 20], Col.PRICE: [2.0, 4.0], Col.BASE_PRICE: [2.5, 4.0]}
        )
        out = LogFeatures().apply(frame)
        np.testing.assert_allclose(out[Col.DISCOUNT_DEPTH], [0.2, 0.0])
        np.testing.assert_allclose(out[Col.LN_UNITS], np.log([10, 20]))


class TestFourierSeasonality:
    """FourierSeasonality adds 2K bounded columns."""

    def test_shape_and_range(self) -> None:
        """Order 3 gives the six named columns, all within [-1, 1]."""
        step = FourierSeasonality(3)
        out = step.apply(
            _weeks([str(d.date()) for d in pd.date_range("2010-01-02", periods=60, freq="7D")])
        )
        assert step.column_names() == ["sin_1", "cos_1", "sin_2", "cos_2", "sin_3", "cos_3"]
        assert out[step.column_names()].abs().le(1).all().all()


class TestHolidayFlags:
    """HolidayFlags marks the week containing each holiday."""

    def test_super_bowl_date(self) -> None:
        """First Sunday of February."""
        assert HolidayFlags.super_bowl_date(2010) == dt.date(2010, 2, 7)
        assert HolidayFlags.super_bowl_date(2009) == dt.date(2009, 2, 1)

    def test_known_weeks(self) -> None:
        """Thanksgiving 2009 (Nov 26) and Super Bowl 2010 (Feb 7) weeks are flagged."""
        out = HolidayFlags("US").apply(_weeks(["2009-11-28", "2010-02-13", "2010-03-20"]))
        assert out[Col.THANKSGIVING].tolist() == [1, 0, 0]
        assert out[Col.SUPER_BOWL].tolist() == [0, 1, 0]
        assert out[Col.HOLIDAY_WEEK].tolist() == [1, 0, 0]  # Super Bowl is not a public holiday

    def test_keeps_row_order(self) -> None:
        """Output rows align with input rows."""
        frame = _weeks(["2010-03-20", "2009-11-28"])
        assert HolidayFlags("US").apply(frame)[Col.WEEK].tolist() == frame[Col.WEEK].tolist()


class TestTimeSplit:
    """TimeSplit holds out the last N weeks globally."""

    def test_no_future_rows_in_train(self) -> None:
        """Every train week precedes every test week; holdout size is exact."""
        frame = _weeks([str(d.date()) for d in pd.date_range("2010-01-02", periods=20, freq="7D")])
        out = TimeSplit(5).apply(frame)
        train, test = (out.loc[out[Col.SPLIT] == s, Col.WEEK] for s in (Split.TRAIN, Split.TEST))
        assert train.max() < test.min() and test.nunique() == 5
        assert out[Col.T].tolist() == list(range(20))

    def test_rejects_oversized_holdout(self) -> None:
        """Holdout covering all weeks is a ConfigError."""
        with pytest.raises(ConfigError):
            TimeSplit(4).apply(_weeks(["2010-01-02", "2010-01-09", "2010-01-16", "2010-01-23"]))


class TestFeatureBuilder:
    """FeatureBuilder applies every registered step."""

    def test_applies_registered_steps(self, cfg: Config) -> None:
        """Output has columns from each registered step."""
        frame = pd.DataFrame(
            {
                Col.WEEK: pd.date_range("2010-01-02", periods=30, freq="7D"),
                Col.UNITS: 5,
                Col.PRICE: 1.0,
                Col.BASE_PRICE: 1.2,
            }  # fmt: skip
        )
        out = FeatureBuilder([s.from_config(cfg) for s in FEATURE_STEPS]).build(frame)
        assert {Col.LN_UNITS, Col.HOLIDAY_WEEK, Col.SPLIT, "sin_1"} <= set(out.columns)
        assert len(out) == len(frame)
