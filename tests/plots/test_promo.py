"""Smoke tests for src.plots.promo.PromoPlotter.

Builds tiny backtest, frontier, forecast and summary tables by hand so all five charts render
for one objective. One test class per production class.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.plots import PlotStyle, PromoPlotter
from src.schema import Col, PromoScenario

matplotlib.use("Agg")

OBJECTIVE = "profit"


def _frontier() -> pd.DataFrame:
    """Four budgets, the third recommended."""
    return pd.DataFrame(
        {Col.BUDGET: [0, 10, 20, 30], Col.GAIN: [0, 8, 12, 13],
         Col.MARGINAL_RETURN: [np.nan, 0.8, 0.4, 0.1],
         Col.RECOMMENDED_BUDGET: [False, False, True, False], Col.OBJECTIVE: OBJECTIVE}
    )  # fmt: skip


def _forecast() -> pd.DataFrame:
    """Two products x three weeks; product 1 has a feature promo in week 2."""
    frame = pd.DataFrame(
        {Col.UPC: [1, 1, 1, 2, 2, 2], Col.HORIZON_WEEK: [1, 2, 3] * 2,
         Col.WEEK: list(pd.date_range("2012-01-11", periods=3, freq="7D")) * 2,
         Col.CATEGORY: "COLD CEREAL", Col.DESCRIPTION: ["A"] * 3 + ["B"] * 3,
         Col.PRODUCT_SIZE: "12 OZ", Col.FORECAST_CURRENT: 100.0, Col.FORECAST_BASELINE: 95.0,
         Col.TACTIC: [None, "feature", None, None, None, None],
         Col.DISCOUNT_DEPTH: [np.nan, 0.1, np.nan, np.nan, np.nan, np.nan],
         Col.OBJECTIVE: OBJECTIVE}
    )  # fmt: skip
    return frame.assign(**{Col.PLAN_UNITS: frame[Col.FORECAST_BASELINE] * 1.2})


def _summary() -> pd.DataFrame:
    """One row per scenario."""
    n = len(PromoScenario)
    data = {f"{m.value}_change": np.linspace(0, 0.2, n) for m in (Col.VOLUME, Col.REVENUE,
                                                                  Col.PROFIT)}  # fmt: skip
    return pd.DataFrame({Col.SCENARIO: [s.value for s in PromoScenario], Col.TRADE_SPEND: 1.0,
                         Col.N_PROMOS: 2, Col.OBJECTIVE: OBJECTIVE, **data})  # fmt: skip


class TestPromoPlotter:
    """All five charts render."""

    def test_backtest(self) -> None:
        """One bar per method."""
        scores = pd.DataFrame({Col.METHOD: ["naive_flat", "xgboost"], Col.WAPE: [0.2, 0.15],
                               Col.BIAS: [0.01, -0.02], Col.SELECTED: [False, True]})  # fmt: skip
        fig = PromoPlotter(PlotStyle()).backtest(scores)
        assert len(fig.axes[0].patches) == 2

    def test_frontier(self) -> None:
        """Two panels: gain curve and marginal return."""
        fig = PromoPlotter(PlotStyle()).frontier(_frontier(), OBJECTIVE, 25.0, 0.25)
        assert isinstance(fig, Figure) and len(fig.axes) == 2

    def test_calendar_only_promoted(self) -> None:
        """Only the promoted product gets a row."""
        fig = PromoPlotter(PlotStyle()).calendar(_forecast(), OBJECTIVE)
        assert len(fig.axes[0].get_yticklabels()) == 1

    def test_scenarios_and_forecast(self) -> None:
        """Scenario bars and the weekly forecast render."""
        plotter = PromoPlotter(PlotStyle())
        assert isinstance(plotter.scenarios(_summary(), OBJECTIVE), Figure)
        assert isinstance(plotter.forecast(_forecast(), OBJECTIVE), Figure)
