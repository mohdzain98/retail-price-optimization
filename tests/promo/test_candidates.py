"""Tests for src.promo.candidates: execution levels, tactic effects, option economics.

Economics are checked against hand-computed values and the accounting identity
incremental profit = incremental revenue - cost x incremental units - fixed fee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.promo.candidates import CandidateScorer, EffectEstimator, ExecutionLevels, PromoEffects
from src.schema import Col, Tactic
from tests.promo.conftest import UNKNOWN_UPC, LinearLift


class TestExecutionLevels:
    """ExecutionLevels takes the median share over weeks the tactic ran."""

    def test_median_when_on(self) -> None:
        """Shares below the threshold are ignored; combined tactics need all columns on."""
        panel = pd.DataFrame(
            {Col.DISPLAY: [0.0, 0.3, 0.5, 0.9], Col.FEATURE: [0.0, 1.0, 0.0, 1.0],
             Col.FEATURE_DISPLAY: [0.0, 0.3, 0.0, 0.8]}
        )  # fmt: skip
        levels = ExecutionLevels([Tactic.DISPLAY, Tactic.FEATURE_DISPLAY], 0.25).estimate(panel)
        assert levels[Tactic.DISPLAY] == {Col.DISPLAY: pytest.approx(0.5)}
        assert levels[Tactic.FEATURE_DISPLAY][Col.DISPLAY] == pytest.approx(0.6)


class TestEffectEstimator:
    """EffectEstimator reads clipped log effects at the execution level."""

    def test_effects(self, lift_model: LinearLift) -> None:
        """display 0.5 x 0.4; TPR's negative coef floored; pl2's positive coef capped."""
        template = pd.DataFrame({Col.UPC: [1, UNKNOWN_UPC], Col.DISPLAY: 0.0, Col.TPR: 0.0})
        levels = {Tactic.DISPLAY: {Col.DISPLAY: 0.4}, Tactic.TPR: {Col.TPR: 1.0}}
        effects = EffectEstimator(lift_model, pantry_lags=2).estimate(template, levels)
        assert effects.tactic.loc[1, Tactic.DISPLAY.value] == pytest.approx(0.2)
        assert effects.tactic.loc[1, Tactic.TPR.value] == 0.0
        assert effects.pantry.loc[1].tolist() == pytest.approx([-0.1, 0.0])
        assert (effects.tactic.loc[UNKNOWN_UPC] == 0).all()


class TestCandidateScorer:
    """CandidateScorer prices every option over the horizon."""

    PRICE, COST, DELTA, EFFECT, DEPTH, FEE = 10.0, 7.0, -2.0, 0.5, 0.2, 300.0

    def _score(self, effective_cost: float) -> pd.DataFrame:
        """One product, 3 weeks of baseline 100/200/300, display at 20%, pl1 = -10%."""
        forecast = pd.DataFrame({Col.UPC: 1, Col.HORIZON_WEEK: [1, 2, 3],
                                 Col.FORECAST_BASELINE: [100.0, 200.0, 300.0]})  # fmt: skip
        products = pd.DataFrame({Col.UPC: [1], Col.REGULAR_PRICE: self.PRICE,
                                 Col.UNIT_COST: self.COST, Col.EFFECTIVE_COST: effective_cost,
                                 Col.PROMO_ELASTICITY: self.DELTA})  # fmt: skip
        effects = PromoEffects(
            pd.DataFrame({Tactic.DISPLAY.value: [self.EFFECT]}, index=[1]),
            pd.DataFrame({"pl1": [np.log(0.9)]}, index=[1]),
        )
        scorer = CandidateScorer([Tactic.DISPLAY], [self.DEPTH], {Tactic.DISPLAY: self.FEE})
        return scorer.score(forecast, products, effects).set_index(Col.HORIZON_WEEK)

    def test_hand_computed_week(self) -> None:
        """Week 1: lift, promo units, dip from week 2, spend and profit by hand."""
        row = self._score(self.COST).loc[1]
        lift = (1 - self.DEPTH) ** self.DELTA * np.exp(self.EFFECT)
        units, dip = 100 * lift, 200 * 0.1
        assert row[Col.LIFT] == pytest.approx(lift)
        assert row[Col.DIP_UNITS] == pytest.approx(dip)
        assert row[Col.TRADE_SPEND] == pytest.approx(self.DEPTH * self.PRICE * units + self.FEE)
        profit = (8.0 - self.COST) * units - (self.PRICE - self.COST) * (100 + dip) - self.FEE
        assert row[Col.INCREMENTAL_PROFIT] == pytest.approx(profit)
        assert row[Col.GAIN] == pytest.approx(profit)

    def test_dip_past_horizon_reuses_last_week(self) -> None:
        """A promo in the last week loses 10% of that week's baseline afterwards."""
        assert self._score(self.COST).loc[3, Col.DIP_UNITS] == pytest.approx(30.0)

    def test_identity_and_revenue_gain(self) -> None:
        """Profit = revenue - c x units - fee; the revenue goal's gain is the revenue."""
        frame = self._score(0.0)
        expected = (frame[Col.INCREMENTAL_REVENUE] - self.COST * frame[Col.INCREMENTAL]
                    - frame[Col.FIXED_COST])  # fmt: skip
        np.testing.assert_allclose(frame[Col.INCREMENTAL_PROFIT], expected)
        np.testing.assert_allclose(frame[Col.GAIN], frame[Col.INCREMENTAL_REVENUE])
