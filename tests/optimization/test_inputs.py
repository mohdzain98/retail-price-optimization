"""Tests for src.optimization.inputs: current state, cost and pack-size parsing.

P0 is the median base price and B the mean baseline over the last N weeks; a
product with no recent sales is excluded. One test class per production class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.optimization.inputs import InputsBuilder, PackSizeParser
from src.schema import Col
from tests.optimization.conftest import make_inputs


class TestPackSizeParser:
    """PackSizeParser normalises sizes to value + unit."""

    def test_units(self) -> None:
        """Litres become millilitres; unparseable sizes give NaN."""
        out = PackSizeParser().parse(pd.Series(["12.25 OZ", "1.5 LT", "16 CT", "family pack"]))
        assert out[Col.SIZE_VALUE].iloc[:3].tolist() == [12.25, 1500.0, 16.0]
        assert out[Col.SIZE_UNIT].iloc[:3].tolist() == ["OZ", "ML", "CT"]
        assert np.isnan(out[Col.SIZE_VALUE].iloc[3])


class TestInputsBuilder:
    """InputsBuilder derives the current state from recent weeks."""

    def test_current_state(self, baseline_and_final: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """P0 = median recent base price, B = mean recent baseline, cost from margin."""
        baseline, final = baseline_and_final
        cfg = Config.load()
        inputs = InputsBuilder(13, cfg.cost, PackSizeParser()).build(baseline, final)
        products = inputs.products.set_index(Col.UPC)
        assert list(products.index) == [1, 2]  # product 3 has no recent sales
        assert products.loc[1, Col.CURRENT_PRICE] == pytest.approx(2.0)
        assert products.loc[1, Col.BASE_VOLUME] == pytest.approx(np.mean(np.arange(7, 20)) + 100)
        margin = cfg.cost.margin_for("BAG SNACKS")
        assert products.loc[2, Col.UNIT_COST] == pytest.approx((1 - margin) * 3.0)


class TestOptimizationInputs:
    """OptimizationInputs exposes aligned arrays and beta scenarios."""

    def test_with_beta(self) -> None:
        """with_beta replaces elasticities without touching the original."""
        inputs = make_inputs([-2.0, -1.5])
        changed = inputs.with_beta(np.array([-3.0, -3.0]))
        assert changed.beta.tolist() == [-3.0, -3.0] and inputs.beta.tolist() == [-2.0, -1.5]
