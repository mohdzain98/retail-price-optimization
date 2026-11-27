"""Tests for src.elasticity.guardrails: quality gates, model selection, imputation.

Uses small hand-built frames with known answers: which model wins per product,
and which hierarchy level an imputed value comes from.
One test class per production class.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import Config, ElasticityConfig, ModelName, QualityConfig
from src.elasticity.base import DesignSpec
from src.elasticity.frequentist import OlsPerProduct
from src.elasticity.guardrails import HierarchicalImputer, ModelSelector, QualityGates
from src.schema import Col, ElasticitySource, Family, ImputeReason
from src.utils import DataValidationError

BOUNDS = {Family.REGULAR: (-5.0, -0.3), Family.PROMO: (-8.0, -0.3)}
LEVELS = [[Col.SUB_CATEGORY, Col.MANUFACTURER], [Col.SUB_CATEGORY], [Col.CATEGORY]]


def _products() -> pd.DataFrame:
    """Five products: two sub-categories in one category; the last is ineligible."""
    return pd.DataFrame(
        {
            Col.UPC: [1, 2, 3, 4, 5],
            Col.DESCRIPTION: list("abcde"),
            Col.PRODUCT_SIZE: "1 OZ",
            Col.MANUFACTURER: ["M1", "M1", "M2", "M1", "M1"],
            Col.CATEGORY: "C",
            Col.SUB_CATEGORY: ["S1", "S1", "S1", "S2", "S1"],
            Col.ELIGIBLE: [True, True, True, True, False],
        }
    )


def _selected(regular: dict[int, float], promo: dict[int, float]) -> pd.DataFrame:
    """Selected rows (model chosen) with +/-0.2 intervals."""
    upcs = list(regular)
    return pd.DataFrame(
        {
            Col.UPC: upcs,
            Col.MODEL: "m",
            Col.ELASTICITY: [regular[u] for u in upcs],
            Col.ELASTICITY_LOW: [regular[u] - 0.2 for u in upcs],
            Col.ELASTICITY_HIGH: [regular[u] + 0.2 for u in upcs],
            Col.PROMO_ELASTICITY: [promo[u] for u in upcs],
            Col.PROMO_ELASTICITY_LOW: [promo[u] - 0.2 for u in upcs],
            Col.PROMO_ELASTICITY_HIGH: [promo[u] + 0.2 for u in upcs],
        }
    )


class TestQualityGates:
    """QualityGates scores fit per product and applies the thresholds."""

    def test_good_fit_passes_and_strict_gates_fail(
        self, spec: DesignSpec, settings: ElasticityConfig, train: pd.DataFrame
    ) -> None:
        """Clean synthetic fits pass; an impossible R^2 threshold fails everything."""
        model = OlsPerProduct(spec, settings)
        est = model.fit(train)
        scored = QualityGates(settings.quality, settings.bounds).evaluate(model, est, train)
        assert scored[Col.PASSES_GATES].all() and (scored[Col.MAPE] < 0.1).all()
        strict = QualityGates(QualityConfig(max_mape=0.3, min_r2=1.01), settings.bounds)
        assert not strict.evaluate(model, est, train)[Col.PASSES_GATES].any()


class TestModelSelector:
    """ModelSelector takes the first passing model in priority order."""

    def test_priority_order(self) -> None:
        """UPC 1 passes both -> first model; UPC 2 only the second; UPC 3 none."""
        a = pd.DataFrame({Col.UPC: [1, 2, 3], Col.PASSES_GATES: [True, False, False]})
        b = pd.DataFrame({Col.UPC: [1, 2, 3], Col.PASSES_GATES: [True, True, False]})
        first, second = ModelName.BAYES_HIERARCHICAL, ModelName.MIXED_EFFECTS
        chosen = ModelSelector([first, second]).select({first: a, second: b})
        by_upc = chosen.set_index(Col.UPC)[Col.MODEL]
        assert by_upc.to_dict() == {1: first.value, 2: second.value}


class TestHierarchicalImputer:
    """HierarchicalImputer fills from the most specific level with valid estimates."""

    def test_levels_and_reasons(self) -> None:
        """UPC 2 (out of bounds) takes its S1+M1 peer; 4 (no model) takes category median."""
        selected = _selected({1: -1.0, 2: 0.4, 3: -2.0}, {1: -2.0, 2: -3.0, 3: -9.5})
        final = HierarchicalImputer(LEVELS, BOUNDS).apply(selected, _products()).set_index(Col.UPC)
        assert final.loc[2, Col.IMPUTE_REASON] == ImputeReason.OUT_OF_BOUNDS
        assert final.loc[2, Col.ELASTICITY] == -1.0  # S1 + M1 peer is UPC 1
        assert final.loc[4, Col.IMPUTE_REASON] == ImputeReason.NO_MODEL_PASSED
        assert final.loc[4, Col.ELASTICITY] == -1.5  # category C median of 1 and 3
        assert final.loc[5, Col.IMPUTE_REASON] == ImputeReason.INELIGIBLE
        assert final.loc[3, Col.PROMO_IMPUTE_REASON] == ImputeReason.OUT_OF_BOUNDS
        assert final.loc[3, Col.ELASTICITY_SOURCE] == ElasticitySource.ESTIMATED
        assert final.loc[3, Col.PROMO_SOURCE] == ElasticitySource.IMPUTED

    def test_no_valid_estimates_raises(self) -> None:
        """Nothing to impute from is an error, not silent NaNs."""
        selected = _selected({1: 1.0}, {1: -2.0})
        with pytest.raises(DataValidationError):
            HierarchicalImputer(LEVELS, BOUNDS).apply(selected, _products())


def test_config_levels_are_columns() -> None:
    """Configured imputation levels are canonical columns."""
    levels = Config.load().elasticity.imputation_levels
    assert all(isinstance(c, Col) for level in levels for c in level)
