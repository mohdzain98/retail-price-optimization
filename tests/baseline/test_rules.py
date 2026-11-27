"""Tests for src.baseline.rules: each RGMx rule and the corrector's precedence.

Small hand-built series make each rule fire (or not) on a known week; the
corrector must prefer rolling-mean fixes over actual-volume fixes.
One test class per production class.
"""

from __future__ import annotations

import pandas as pd

from src.baseline.rules import (
    HOLIDAY,
    NEXT,
    PREV,
    RECENT_PROMO,
    BaselineCorrector,
    ImplausiblyHigh,
    PromoDip,
    PromoDrop,
    PromoSpike,
    ZeroBaseline,
)
from src.schema import BaselineFix, Col


def _view(base: list[float], units: list[float], promo: list[int], **extra: list) -> pd.DataFrame:
    """Rule view for one product with neighbour baselines precomputed."""
    frame = pd.DataFrame({Col.BASE_RAW: base, Col.UNITS: units, Col.PROMO_FLAG: promo})
    frame[PREV], frame[NEXT] = frame[Col.BASE_RAW].shift(1), frame[Col.BASE_RAW].shift(-1)
    frame[RECENT_PROMO] = extra.get("recent", [False] * len(base))
    frame[HOLIDAY] = extra.get("holiday", [False] * len(base))
    return frame


class TestZeroBaseline:
    """Rule 2."""

    def test_flags_zero_with_sales(self) -> None:
        """Zero baseline with positive units is flagged."""
        view = _view([0.0, 5.0], [3.0, 5.0], [0, 0])
        assert ZeroBaseline().flags(view).tolist() == [True, False]


class TestImplausiblyHigh:
    """Rule 3."""

    def test_ratio_and_no_recent_promo(self) -> None:
        """Ratio > 1.5 flags; base > units flags only without a recent promo."""
        view = _view([20.0, 11.0, 11.0], [10.0, 10.0, 10.0], [0, 0, 0], recent=[False, False, True])
        assert ImplausiblyHigh(1.5).flags(view).tolist() == [True, True, False]


class TestPromoDip:
    """Rule 4."""

    def test_promo_week_below_both_neighbours(self) -> None:
        """Only the promo week that dips below both neighbours is flagged."""
        view = _view([10.0, 6.0, 10.0], [10.0, 20.0, 10.0], [0, 1, 0])
        assert PromoDip().flags(view).fillna(False).tolist() == [False, True, False]


class TestPromoSpike:
    """Rule 5."""

    def test_spike_unless_holiday(self) -> None:
        """A promo-week spike is flagged, except in a holiday week."""
        base, units, promo = [10.0, 14.0, 10.0], [10.0, 20.0, 10.0], [0, 1, 0]
        assert PromoSpike().flags(_view(base, units, promo)).fillna(False).tolist()[1]
        holiday = _view(base, units, promo, holiday=[False, True, False])
        assert not PromoSpike().flags(holiday).fillna(False).tolist()[1]


class TestPromoDrop:
    """Rule 6."""

    def test_below_ratio_of_a_neighbour(self) -> None:
        """Promo week below 75% of one neighbour is flagged."""
        view = _view([10.0, 7.0, 6.0], [10.0, 20.0, 6.0], [0, 1, 0])
        assert PromoDrop(0.75).flags(view).fillna(False).tolist() == [False, True, False]


class TestBaselineCorrector:
    """Precedence and outputs of the corrector."""

    def test_rolling_beats_actual(self) -> None:
        """A dip week takes the 3-week rolling mean; others stay raw."""
        frame = pd.DataFrame(
            {
                Col.UPC: 1,
                Col.WEEK: pd.date_range("2010-01-02", periods=3, freq="7D"),
                Col.BASE_RAW: [10.0, 4.0, 10.0],
                Col.UNITS: [10.0, 20.0, 10.0],
                Col.PROMO_FLAG: [0, 1, 0],
            }
        )
        corrector = BaselineCorrector([PromoDip(), ZeroBaseline()], rolling_window=3, lookback=3)
        out = corrector.correct(frame)
        assert out[Col.BASELINE_FIX].tolist() == [
            BaselineFix.NONE.value, BaselineFix.ROLLING.value, BaselineFix.NONE.value,
        ]  # fmt: skip
        assert out[Col.BASELINE].iloc[1] == 8.0  # mean of 10, 4, 10
        assert "rule_promo_dip" in out.columns
