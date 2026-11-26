"""Fixtures for elasticity tests: a product-week panel with known elasticities.

SyntheticPanel builds the modelling columns directly (no prep step) with separate
true regular-price and promo-price elasticities, so each model can be checked
against both. One extra product is ineligible to exercise imputation.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from src.config import Config, ElasticityConfig
from src.elasticity.base import DesignSpec
from src.schema import Col, Family, Split


class SyntheticPanel:
    """ln_units = level + b_reg*(ln base - mean) + b_promo*ln(price/base) + promo + noise.

    Regular prices step every 13 weeks (sd 12%); 25% of weeks carry a 5-35% temporary discount
    with its own TPR lift (0.1), so depth and the TPR flag are separately identified.
    TRUE maps UPC -> (category, regular elasticity, promo elasticity).
    """

    N_WEEKS: ClassVar[int] = 104
    HOLDOUT: ClassVar[int] = 13
    TRUE: ClassVar[dict[int, tuple[str, float, float]]] = {
        101: ("CEREAL", -1.6, -2.6), 102: ("CEREAL", -2.0, -3.0),
        103: ("CEREAL", -2.4, -3.4), 104: ("CEREAL", -1.8, -2.8),
        201: ("SNACKS", -1.0, -2.0), 202: ("SNACKS", -1.4, -2.4),
        203: ("SNACKS", -1.2, -2.2), 204: ("SNACKS", -0.8, -1.8),
    }  # fmt: skip
    INELIGIBLE_UPC: ClassVar[int] = 999

    def __init__(self, seed: int = 7) -> None:
        self._rng = np.random.default_rng(seed)

    def build(self) -> pd.DataFrame:
        """Full panel (train + test) including the ineligible product."""
        frames = [self._product(upc, *truth) for upc, truth in self.TRUE.items()]
        ineligible = self._product(self.INELIGIBLE_UPC, "SNACKS", -1.1, -2.1)
        frames.append(ineligible.assign(**{Col.ELIGIBLE: False}))
        return pd.concat(frames, ignore_index=True)

    def _prices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(ln base price, ln promo price, tpr flag) for one product."""
        n, rng = self.N_WEEKS, self._rng
        steps = np.repeat(rng.normal(0, 0.12, n // 13 + 1), 13)[:n]
        ln_base = np.log(3.0) + steps + rng.normal(0, 0.01, n)
        tpr = rng.random(n) < 0.25
        ln_promo = np.where(tpr, np.log(1 - rng.uniform(0.05, 0.35, n)), 0.0)
        return ln_base, ln_promo, tpr

    def _product(self, upc: int, category: str, beta: float, beta_promo: float) -> pd.DataFrame:
        """One product's weekly rows."""
        n, rng = self.N_WEEKS, self._rng
        t = np.arange(n)
        ln_base, ln_promo, tpr = self._prices()
        feature = (rng.random(n) < 0.1).astype(float)
        display = (rng.random(n) < 0.3).astype(float)
        ln_stores = np.log(60 + rng.integers(-5, 6, n))
        trend, season = 0.002 * t, np.sin(2 * np.pi * t / 52)
        ln_units = (
            5 + trend + beta * (ln_base - ln_base.mean()) + beta_promo * ln_promo
            + 0.3 * feature + 0.2 * display + 0.1 * tpr + (ln_stores - ln_stores.mean())
            + 0.2 * season + rng.normal(0, 0.08, n)
        )  # fmt: skip
        return pd.DataFrame(
            {
                Col.UPC: upc, Col.CATEGORY: category, Col.SUB_CATEGORY: f"{category} SUB",
                Col.MANUFACTURER: "ACME", Col.DESCRIPTION: f"ITEM {upc}",
                Col.PRODUCT_SIZE: "10 OZ", Col.ELIGIBLE: True,
                Col.WEEK: pd.date_range("2010-01-02", periods=n, freq="7D"),
                Col.LN_BASE_PRICE: ln_base, Col.LN_PROMO_PRICE: ln_promo,
                Col.LN_PRICE: ln_base + ln_promo, Col.TPR: tpr.astype(float),
                Col.FEATURE: feature, Col.DISPLAY: display, Col.LN_N_STORES: ln_stores,
                Col.LN_TREND: trend, "sin_1": season, "cos_1": np.cos(2 * np.pi * t / 52),
                Col.LN_UNITS: ln_units, Col.UNITS: np.exp(ln_units),
                Col.SPLIT: np.where(t >= n - self.HOLDOUT, Split.TEST.value, Split.TRAIN.value),
            }
        )  # fmt: skip


@pytest.fixture(scope="session")
def synthetic_panel() -> pd.DataFrame:
    """Full synthetic panel, built once per session."""
    return SyntheticPanel().build()


@pytest.fixture(scope="session")
def train(synthetic_panel: pd.DataFrame) -> pd.DataFrame:
    """Eligible training rows."""
    p = synthetic_panel
    return p[p[Col.ELIGIBLE] & (p[Col.SPLIT] == Split.TRAIN)]


@pytest.fixture(scope="session")
def test_rows(synthetic_panel: pd.DataFrame) -> pd.DataFrame:
    """Eligible holdout rows."""
    p = synthetic_panel
    return p[p[Col.ELIGIBLE] & (p[Col.SPLIT] == Split.TEST)]


@pytest.fixture(scope="session")
def spec() -> DesignSpec:
    """Spec from the shipped config (controls missing from the panel are skipped)."""
    return DesignSpec.from_config(Config.load())


@pytest.fixture(scope="session")
def settings() -> ElasticityConfig:
    """Elasticity settings with a small, fast sampler for tests."""
    base = Config.load().elasticity
    fast = base.bayes.model_copy(update={"draws": 300, "tune": 400, "chains": 2})
    return base.model_copy(update={"bayes": fast})


def true_elasticity(upcs: pd.Series, family: Family = Family.REGULAR) -> np.ndarray:
    """True elasticity for each UPC in order, for one family."""
    position = 1 if family is Family.REGULAR else 2
    return np.array([SyntheticPanel.TRUE[u][position] for u in upcs])
