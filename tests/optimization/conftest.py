"""Fixtures for optimization tests: small product tables with known answers.

`make_inputs` builds OptimizationInputs directly; `baseline_and_final` builds the
weekly baseline and final elasticity tables the InputsBuilder consumes. Products
include a private-label / branded pair so price ladders can be exercised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config, ModelName, OptimizationConfig
from src.optimization.inputs import OptimizationInputs
from src.schema import Col, ElasticitySource


def make_inputs(
    beta: list[float], p0: list[float] | None = None, cost_share: float = 0.6
) -> OptimizationInputs:
    """Inputs for len(beta) products: PL pretzel + branded pretzels by default."""
    n = len(beta)
    p0_arr = np.array(p0 if p0 is not None else [2.0] + [3.0] * (n - 1))
    return OptimizationInputs(
        pd.DataFrame(
            {
                Col.UPC: np.arange(1, n + 1),
                Col.DESCRIPTION: [f"ITEM {i}" for i in range(n)],
                Col.PRODUCT_SIZE: "16 OZ",
                Col.CATEGORY: "SNACKS",
                Col.SUB_CATEGORY: "PRETZELS",
                Col.MANUFACTURER: ["PRIVATE LABEL"] + ["BRAND"] * (n - 1),
                Col.CURRENT_PRICE: p0_arr,
                Col.BASE_VOLUME: 100.0,
                Col.ELASTICITY: beta,
                Col.ELASTICITY_LOW: np.array(beta) - 0.3,
                Col.ELASTICITY_HIGH: np.array(beta) + 0.3,
                Col.ELASTICITY_SOURCE: ElasticitySource.ESTIMATED.value,
                Col.MODEL: ModelName.BAYES_HIERARCHICAL.value,
                Col.UNIT_COST: cost_share * p0_arr,
                Col.SIZE_VALUE: 16.0,
                Col.SIZE_UNIT: "OZ",
            }
        )  # fmt: skip
    )


@pytest.fixture
def opt_cfg() -> OptimizationConfig:
    """Shipped optimization config."""
    return Config.load().optimization


@pytest.fixture
def baseline_and_final() -> tuple[pd.DataFrame, pd.DataFrame]:
    """20 weeks for 3 products; product 3 stops selling before the last 13 weeks."""
    weeks = pd.date_range("2011-01-01", periods=20, freq="7D")
    rows = []
    for upc, price in ((1, 2.0), (2, 3.0), (3, 4.0)):
        for i, week in enumerate(weeks):
            if upc == 3 and i >= 5:
                continue
            base_price = price + (0.1 if i == 19 else 0.0)
            rows.append(
                {Col.UPC: upc, Col.WEEK: week, Col.BASE_PRICE: base_price, Col.BASELINE: 100.0 + i}
            )
    final = pd.DataFrame(
        {
            Col.UPC: [1, 2, 3], Col.DESCRIPTION: ["PL PRETZEL", "BRAND PRETZEL", "OLD ITEM"],
            Col.PRODUCT_SIZE: ["16 OZ", "16 OZ", "1.5 LT"],
            Col.MANUFACTURER: ["PRIVATE LABEL", "B", "B"],
            Col.CATEGORY: "BAG SNACKS", Col.SUB_CATEGORY: "PRETZELS",
            Col.ELASTICITY: [-2.0, -1.5, -1.0], Col.ELASTICITY_LOW: -2.5, Col.ELASTICITY_HIGH: -0.5,
            Col.ELASTICITY_SOURCE: ElasticitySource.ESTIMATED.value,
            Col.MODEL: ModelName.BAYES_HIERARCHICAL.value,
        }
    )  # fmt: skip
    return pd.DataFrame(rows), final
