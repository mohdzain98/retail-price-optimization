"""Fixtures for baseline tests: a stub model with a known linear predictor.

`LinearStub` predicts ln_units = 5 + 0.3*feature - 0.2*pl1 - 2*ln_promo_price, so
every effect term has an exact expected value. `week_panel` is a 6-week panel for
one eligible product and one ineligible product.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config, ElasticityConfig, ModelName
from src.elasticity.base import DesignSpec, ElasticityModel
from src.schema import Col

FINAL_PROMO_ELASTICITY = -2.5
ELIGIBLE_UPC, INELIGIBLE_UPC = 1, 2


class LinearStub(ElasticityModel):
    """Test double with a fixed linear predictor (no fitting)."""

    name = ModelName.OLS_PER_PRODUCT
    INTERCEPT, FEATURE, PANTRY, PROMO_PRICE = 5.0, 0.3, -0.2, -2.0

    def __init__(self) -> None:
        cfg = Config.load()
        super().__init__(DesignSpec(controls=()), cfg.elasticity)

    def fit(self, train: pd.DataFrame) -> pd.DataFrame:
        """Nothing to fit."""
        return pd.DataFrame()

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Known linear predictor; NaN for the ineligible product."""
        linear = (
            self.INTERCEPT + self.FEATURE * frame[Col.FEATURE] + self.PANTRY * frame["pl1"]
            + self.PROMO_PRICE * frame[Col.LN_PROMO_PRICE]
        )  # fmt: skip
        return linear.where(frame[Col.UPC] == ELIGIBLE_UPC)


def _product(upc: int) -> pd.DataFrame:
    """Six weeks: week 2 is a promo week (feature, 20% off), week 3 has pl1=1."""
    promo = [0, 0, 1, 0, 0, 0]
    ln_promo = np.where(np.array(promo) == 1, np.log(0.8), 0.0)
    feature = np.array(promo, dtype=float)
    pl1 = np.roll(promo, 1)
    ln_units = (
        5 + 0.3 * feature - 0.2 * pl1 - 2.0 * ln_promo + np.where(np.array(promo) == 1, 0.1, 0.0)
    )
    return pd.DataFrame(
        {
            Col.UPC: upc, Col.WEEK: pd.date_range("2010-01-02", periods=6, freq="7D"),
            Col.DESCRIPTION: f"ITEM {upc}", Col.PRODUCT_SIZE: "1 OZ", Col.CATEGORY: "C",
            Col.PROMO_FLAG: promo, Col.FEATURE: feature, Col.DISPLAY: 0.0,
            Col.FEATURE_DISPLAY: 0.0, Col.TPR: 0.0, "pl1": pl1, "pl2": 0, "pl3": 0,
            Col.LN_PROMO_PRICE: ln_promo, Col.LN_UNITS: ln_units, Col.UNITS: np.exp(ln_units),
            Col.HOLIDAY_WEEK: 0, Col.SUPER_BOWL: 0, Col.THANKSGIVING: 0, Col.CHRISTMAS: 0,
        }
    )  # fmt: skip


@pytest.fixture
def week_panel() -> pd.DataFrame:
    """Panel with one eligible and one ineligible product."""
    return pd.concat([_product(ELIGIBLE_UPC), _product(INELIGIBLE_UPC)], ignore_index=True)


@pytest.fixture
def final_table() -> pd.DataFrame:
    """Final elasticity table: product 1 uses the stub, product 2 is ineligible."""
    return pd.DataFrame(
        {
            Col.UPC: [ELIGIBLE_UPC, INELIGIBLE_UPC],
            Col.MODEL: [ModelName.OLS_PER_PRODUCT.value, None],
            Col.ELIGIBLE: [True, False],
            Col.PROMO_ELASTICITY: [FINAL_PROMO_ELASTICITY, FINAL_PROMO_ELASTICITY],
        }
    )


@pytest.fixture
def settings() -> ElasticityConfig:
    """Shipped elasticity settings."""
    return Config.load().elasticity
