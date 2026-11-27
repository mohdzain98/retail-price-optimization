"""Tests for src.elasticity.bayes: the hierarchical model recovers known slopes.

Uses a short sampler (2 chains x 300 draws) so the test stays fast; recovery
tolerance and HDI coverage are set loosely enough to be stable at that size.
One test class per production class, plus the HDI helper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import ElasticityConfig
from src.elasticity.base import DesignSpec
from src.elasticity.bayes import BayesianHierarchical, _hdi
from src.schema import ESTIMATE_COLS, Col, Family
from tests.elasticity.conftest import true_elasticity


@pytest.fixture(scope="module")
def fitted(
    spec: DesignSpec, settings: ElasticityConfig, train: pd.DataFrame
) -> tuple[BayesianHierarchical, pd.DataFrame]:
    """Fit once per module (sampling is the slow part)."""
    model = BayesianHierarchical(spec, settings)
    return model, model.fit(train)


class TestBayesianHierarchical:
    """Recovery, interval coverage, prediction and artifacts."""

    @pytest.mark.parametrize("family", list(Family))
    def test_recovers_true_elasticity(
        self, family: Family, fitted: tuple[BayesianHierarchical, pd.DataFrame]
    ) -> None:
        """Posterior means are near the truth and most HDIs contain it."""
        _, est = fitted
        value, low, high = ESTIMATE_COLS[family]
        truth = true_elasticity(est[Col.UPC], family)
        assert np.abs(est[value] - truth).max() < 0.35
        covered = (est[low] <= truth) & (truth <= est[high])
        assert covered.mean() >= 0.75

    def test_predicts_holdout(
        self, fitted: tuple[BayesianHierarchical, pd.DataFrame], test_rows: pd.DataFrame
    ) -> None:
        """Posterior-mean predictions track the holdout."""
        model, _ = fitted
        error = model.predict(test_rows) - test_rows[Col.LN_UNITS]
        assert float(np.sqrt((error**2).mean())) < 0.2

    def test_saves_draws(
        self, fitted: tuple[BayesianHierarchical, pd.DataFrame], tmp_path: Path
    ) -> None:
        """Posterior draws hold both families, one row per draw and product."""
        model, est = fitted
        model.save_artifacts(tmp_path)
        draws = pd.read_parquet(tmp_path / "elasticity_draws.parquet")
        assert set(draws[Col.UPC]) == set(est[Col.UPC])
        assert {Col.ELASTICITY, Col.PROMO_ELASTICITY} <= set(draws.columns)
        assert len(draws) == 2 * 300 * len(est)

    def test_hdi_reference(self) -> None:
        """HDI of evenly spaced samples 0..100 at 90% spans 90 units."""
        low, high = _hdi(np.arange(101, dtype=float), 0.9)
        assert high - low == pytest.approx(90.0)
