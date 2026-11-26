"""Tests for src.elasticity.base: spec, price centring and holdout scoring.

A tiny oracle model (predicts the true ln_units) pins WAPE at zero, so the
scorer is checked against a known reference value.
One test class per production class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config, ElasticityConfig, ModelName
from src.elasticity.base import (
    SLOPE_COLS,
    DesignSpec,
    ElasticityModel,
    HoldoutScorer,
    PriceCentering,
)
from src.schema import Col, Family
from src.utils import DataValidationError


class _Oracle(ElasticityModel):
    """Test double that predicts the observed ln_units (or NaN for one UPC)."""

    name = ModelName.OLS_PER_PRODUCT

    def __init__(
        self, spec: DesignSpec, settings: ElasticityConfig, hide_upc: int | None = None
    ) -> None:
        super().__init__(spec, settings)
        self._hide = hide_upc

    def fit(self, train: pd.DataFrame) -> pd.DataFrame:
        """Nothing to fit."""
        return pd.DataFrame()

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Return the truth, masking one product."""
        return frame[Col.LN_UNITS].where(frame[Col.UPC] != self._hide)


class TestDesignSpec:
    """DesignSpec lists controls and keeps only usable ones."""

    def test_from_config_controls(self) -> None:
        """Tactics (incl. TPR), pantry lags, Fourier and trend are controls; prices are not."""
        controls = DesignSpec.from_config(Config.load()).controls
        assert {Col.TPR, Col.FEATURE_DISPLAY, "pl1", "sin_1", Col.LN_TREND} <= set(controls)
        assert Col.LN_BASE_PRICE not in controls and Col.LN_PROMO_PRICE not in controls

    def test_active_controls_drop_missing_and_constant(self) -> None:
        """Absent and zero-variance columns are excluded."""
        spec = DesignSpec(controls=(Col.FEATURE, Col.DISPLAY, "sin_1"))
        frame = pd.DataFrame({Col.FEATURE: [0, 1, 0], Col.DISPLAY: [1, 1, 1]})
        assert spec.active_controls(frame) == [Col.FEATURE]


class TestPriceCentering:
    """PriceCentering centres base price per product and passes the promo term through."""

    def test_slopes(self, train: pd.DataFrame) -> None:
        """Regular slope averages zero per product; promo slope equals ln(price/base)."""
        slopes = PriceCentering().fit(train).transform(train)
        regular, promo = SLOPE_COLS[Family.REGULAR], SLOPE_COLS[Family.PROMO]
        assert np.allclose(slopes[regular].groupby(train[Col.UPC]).mean(), 0)
        assert np.allclose(slopes[promo], train[Col.LN_PROMO_PRICE])

    def test_unseen_and_unfitted(self, train: pd.DataFrame) -> None:
        """Unseen products get NaN; transform before fit fails loudly."""
        centering = PriceCentering().fit(train)
        unseen = train.head(1).assign(**{Col.UPC: -1})
        assert centering.transform(unseen)[SLOPE_COLS[Family.REGULAR]].isna().all()
        with pytest.raises(DataValidationError):
            PriceCentering().transform(train)


class TestHoldoutScorer:
    """HoldoutScorer computes WAPE over predictable rows."""

    def test_oracle_scores_zero(
        self, spec: DesignSpec, settings: ElasticityConfig, test_rows: pd.DataFrame
    ) -> None:
        """Perfect predictions give WAPE 0 on every row."""
        score = HoldoutScorer().score(_Oracle(spec, settings), test_rows)
        assert score.wape == pytest.approx(0.0) and score.n_rows == len(test_rows)

    def test_excludes_unpredictable_rows(
        self, spec: DesignSpec, settings: ElasticityConfig, test_rows: pd.DataFrame
    ) -> None:
        """Rows with NaN predictions are dropped from the count."""
        hidden = int(test_rows[Col.UPC].iloc[0])
        score = HoldoutScorer().score(_Oracle(spec, settings, hide_upc=hidden), test_rows)
        assert score.n_rows == int((test_rows[Col.UPC] != hidden).sum())
