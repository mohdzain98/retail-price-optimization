"""Fixtures for promo tests: a linear stub lift model and candidate-table builders.

`LinearLift` predicts ln_units from known tactic, pantry and seasonal coefficients,
so effects and indices have exact expected values. `make_candidates` builds option
tables directly for planner and frontier tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config, ModelName, Objective
from src.elasticity.base import DesignSpec, ElasticityModel
from src.schema import Col, Metric, Scenario, Tactic, kpi_col

UNKNOWN_UPC = 9999  # the stub cannot predict it (like an ineligible product)


class LinearLift(ElasticityModel):
    """Test double: ln_units = 5 + sum(coef * column); NaN for UNKNOWN_UPC."""

    name = ModelName.BAYES_HIERARCHICAL
    COEFS: dict[str, float] = {
        Col.FEATURE: 0.2, Col.DISPLAY: 0.5, Col.FEATURE_DISPLAY: 0.1, Col.TPR: -0.1,
        "pl1": -0.1, "pl2": 0.05, "sin_1": 0.3, Col.HOLIDAY_WEEK: 0.2,
    }  # fmt: skip

    def __init__(self) -> None:
        super().__init__(DesignSpec(controls=()), Config.load().elasticity)

    def fit(self, train: pd.DataFrame) -> pd.DataFrame:
        """Nothing to fit."""
        return pd.DataFrame()

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Known linear predictor over the columns present."""
        linear = pd.Series(5.0, index=frame.index)
        for col, coef in self.COEFS.items():
            if col in frame:
                linear = linear + coef * frame[col].astype(float)
        return linear.where(frame[Col.UPC] != UNKNOWN_UPC)


@pytest.fixture
def lift_model() -> LinearLift:
    """The stub lift model."""
    return LinearLift()


def make_candidates(rows: list[tuple]) -> pd.DataFrame:
    """Options from (upc, horizon_week, tactic, gain, spend) tuples; depth 10%."""
    frame = pd.DataFrame(
        rows, columns=[Col.UPC, Col.HORIZON_WEEK, Col.TACTIC, Col.GAIN, Col.TRADE_SPEND]
    )
    return frame.assign(**{Col.DISCOUNT_DEPTH: 0.1})


def random_candidates(seed: int = 0, n_upc: int = 6, weeks: int = 13) -> pd.DataFrame:
    """Every upc x week x tactic x depth with random gains (some negative) and spends."""
    rng = np.random.default_rng(seed)
    grid = pd.MultiIndex.from_product(
        [range(1, n_upc + 1), range(1, weeks + 1), [t.value for t in Tactic], [0.1, 0.2, 0.3]],
        names=[Col.UPC, Col.HORIZON_WEEK, Col.TACTIC, Col.DISCOUNT_DEPTH],
    ).to_frame(index=False)
    return grid.assign(
        **{
            Col.GAIN: rng.normal(100, 150, len(grid)),
            Col.TRADE_SPEND: rng.uniform(200, 2000, len(grid)),
        }
    )


def make_recommendations(panel: pd.DataFrame) -> pd.DataFrame:
    """Phase 4-style table for the panel's products, profit and revenue blocks."""
    last = panel.sort_values(Col.WEEK).groupby(Col.UPC).tail(13)
    state = last.groupby(Col.UPC).agg(
        **{Col.CURRENT_PRICE: (Col.BASE_PRICE, "median"), "volume": (Col.UNITS, "mean")}
    )
    attrs = panel.drop_duplicates(Col.UPC).set_index(Col.UPC)
    base = state.assign(
        **{
            Col.DESCRIPTION: attrs[Col.DESCRIPTION], Col.PRODUCT_SIZE: attrs[Col.PRODUCT_SIZE],
            Col.CATEGORY: attrs[Col.CATEGORY], Col.UNIT_COST: 0.7 * state[Col.CURRENT_PRICE],
            kpi_col(Scenario.CURRENT, Metric.VOLUME): state["volume"],
            kpi_col(Scenario.RECOMMENDED, Metric.PRICE): 1.05 * state[Col.CURRENT_PRICE],
            kpi_col(Scenario.RECOMMENDED, Metric.VOLUME): state["volume"] * 1.05**-1.5,
        }
    ).drop(columns="volume").reset_index()  # fmt: skip
    return pd.concat(
        [
            base.assign(**{Col.OBJECTIVE: Objective.PROFIT.value,
                           Col.EFFECTIVE_COST: base[Col.UNIT_COST]}),
            base.assign(**{Col.OBJECTIVE: Objective.REVENUE.value, Col.EFFECTIVE_COST: 0.0}),
        ],
        ignore_index=True,
    )  # fmt: skip
