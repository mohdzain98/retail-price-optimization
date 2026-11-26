"""Tests for src.elasticity.frequentist: each model recovers known elasticities.

The synthetic panel has true regular and promo slopes per product; per-product
OLS and MixedLM must recover both, pooled OLS each category's average.
One test class per production class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import ElasticityConfig
from src.elasticity.base import ESTIMATE_FRAME_COLS, DesignSpec
from src.elasticity.frequentist import MixedEffects, OlsPerProduct, PooledOls
from src.schema import ESTIMATE_COLS, Col, Family
from tests.elasticity.conftest import SyntheticPanel, true_elasticity

TOLERANCE = {Family.REGULAR: 0.35, Family.PROMO: 0.35}


def _max_error(est: pd.DataFrame, family: Family) -> float:
    """Largest absolute gap between estimates and truth for one family."""
    column = ESTIMATE_COLS[family][0]
    return float(np.abs(est[column] - true_elasticity(est[Col.UPC], family)).max())


class TestOlsPerProduct:
    """Per-product OLS recovers both slopes and predicts the holdout."""

    @pytest.mark.parametrize("family", list(Family))
    def test_recovers_true_elasticity(
        self, family: Family, spec: DesignSpec, settings: ElasticityConfig, train: pd.DataFrame
    ) -> None:
        """Every estimate is within tolerance; intervals bracket the estimate."""
        est = OlsPerProduct(spec, settings).fit(train)
        assert _max_error(est, family) < TOLERANCE[family]
        value, low, high = ESTIMATE_COLS[family]
        assert ((est[low] < est[value]) & (est[value] < est[high])).all()

    def test_predicts_holdout(
        self,
        spec: DesignSpec,
        settings: ElasticityConfig,
        train: pd.DataFrame,
        test_rows: pd.DataFrame,
    ) -> None:
        """Holdout predictions are close to the truth (noise sd is 0.08)."""
        model = OlsPerProduct(spec, settings)
        model.fit(train)
        error = model.predict(test_rows) - test_rows[Col.LN_UNITS]
        assert float(np.sqrt((error**2).mean())) < 0.2


class TestPooledOls:
    """Pooled OLS gives every product its category's slopes."""

    @pytest.mark.parametrize("family", list(Family))
    def test_recovers_category_average(
        self, family: Family, spec: DesignSpec, settings: ElasticityConfig, train: pd.DataFrame
    ) -> None:
        """Category slope is close to the mean true elasticity in that category."""
        est = PooledOls(spec, settings).fit(train)
        column = ESTIMATE_COLS[family][0]
        category = np.array([SyntheticPanel.TRUE[u][0] for u in est[Col.UPC]])
        truth = pd.Series(true_elasticity(est[Col.UPC], family)).groupby(category).mean()
        fitted = est.groupby(category)[column].first()
        assert (fitted - truth).abs().max() < TOLERANCE[family]
        assert est.groupby(category)[column].nunique().eq(1).all()

    def test_unseen_product_is_nan(
        self, spec: DesignSpec, settings: ElasticityConfig, train: pd.DataFrame
    ) -> None:
        """Products not in training cannot be predicted."""
        model = PooledOls(spec, settings)
        model.fit(train)
        assert model.predict(train.head(2).assign(**{Col.UPC: -1})).isna().all()


class TestMixedEffects:
    """MixedLM recovers product slopes via partial pooling."""

    def test_recovers_true_elasticity(
        self,
        spec: DesignSpec,
        settings: ElasticityConfig,
        train: pd.DataFrame,
        test_rows: pd.DataFrame,
    ) -> None:
        """Both families within tolerance; predictions finite for fitted products."""
        model = MixedEffects(spec, settings)
        est = model.fit(train)
        assert list(est.columns) == list(ESTIMATE_FRAME_COLS)
        for family in Family:
            assert _max_error(est, family) < TOLERANCE[family]
        assert np.isfinite(model.predict(test_rows)).all()
