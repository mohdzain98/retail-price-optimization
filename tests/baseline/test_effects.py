"""Tests for src.baseline.effects.EffectCalculator against exact stub values.

The stub's predictor is known, so promo, pantry, price and error effects and the
raw baseline can be checked to machine precision, including the ineligible case.
One test class per production class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baseline.effects import EffectCalculator
from src.config import ModelName
from src.schema import EFFECT_COLS, Col, EffectTerm
from tests.baseline.conftest import (
    ELIGIBLE_UPC,
    FINAL_PROMO_ELASTICITY,
    INELIGIBLE_UPC,
    LinearStub,
)

PROMO_WEEK, POST_PROMO_WEEK = 2, 3


@pytest.fixture
def effects(week_panel: pd.DataFrame, final_table: pd.DataFrame) -> pd.DataFrame:
    """Effect terms for the 6-week panel."""
    calculator = EffectCalculator(
        {ModelName.OLS_PER_PRODUCT: LinearStub()}, [ModelName.OLS_PER_PRODUCT], pantry_lags=3
    )
    return calculator.compute(week_panel, final_table)


class TestEffectCalculator:
    """Effect terms follow the RGMx top-down definitions."""

    def test_promo_week_terms(self, effects: pd.DataFrame) -> None:
        """Promo week: tactic 0.3, price = final delta x ln(0.8), error = +0.1 residual."""
        row = effects[effects[Col.UPC] == ELIGIBLE_UPC].iloc[PROMO_WEEK]
        assert row[EFFECT_COLS[EffectTerm.PROMO]] == pytest.approx(0.3)
        assert row[EFFECT_COLS[EffectTerm.PRICE]] == pytest.approx(
            FINAL_PROMO_ELASTICITY * np.log(0.8)
        )
        assert row[EFFECT_COLS[EffectTerm.ERROR]] == pytest.approx(0.1)

    def test_post_promo_pantry_dip_is_restored(self, effects: pd.DataFrame) -> None:
        """Week after the promo: pantry effect -0.2 raises the baseline above actual."""
        row = effects[effects[Col.UPC] == ELIGIBLE_UPC].iloc[POST_PROMO_WEEK]
        assert row[EFFECT_COLS[EffectTerm.PANTRY]] == pytest.approx(-0.2)
        assert row[Col.BASE_RAW] == pytest.approx(row[Col.UNITS] * np.exp(0.2))

    def test_non_promo_weeks_keep_actual(self, effects: pd.DataFrame) -> None:
        """Weeks with no promo and no pantry effect: baseline equals actual units."""
        rows = effects[effects[Col.UPC] == ELIGIBLE_UPC].iloc[[0, 1, 4, 5]]
        np.testing.assert_allclose(rows[Col.BASE_RAW], rows[Col.UNITS])

    def test_ineligible_gets_price_effect_only(self, effects: pd.DataFrame) -> None:
        """No model: promo/pantry/error are 0; price uses the (imputed) final delta."""
        rows = effects[effects[Col.UPC] == INELIGIBLE_UPC]
        assert rows[Col.BASELINE_MODEL].isna().all()
        others = [EFFECT_COLS[t] for t in (EffectTerm.PROMO, EffectTerm.PANTRY, EffectTerm.ERROR)]
        assert (rows[others] == 0).all().all()
        assert rows[EFFECT_COLS[EffectTerm.PRICE]].iloc[PROMO_WEEK] > 0
