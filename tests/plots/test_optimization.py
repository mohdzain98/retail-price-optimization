"""Smoke tests for src.plots.optimization.OptimizationPlotter.

Builds a tiny recommendations table and summary by hand so all three charts render.
One test class per production class.
"""

from __future__ import annotations

import matplotlib
import pandas as pd
from matplotlib.figure import Figure

from src.plots import OptimizationPlotter, PlotStyle
from src.schema import Col, Metric, Scenario, kpi_col

matplotlib.use("Agg")


def _recs() -> pd.DataFrame:
    """Two products in two categories with current and recommended KPIs."""
    frame = pd.DataFrame(
        {
            Col.UPC: [1, 2], Col.CATEGORY: ["A", "B"], Col.DESCRIPTION: ["X", "Y"],
            Col.PRODUCT_SIZE: "1 OZ", Col.CURRENT_PRICE: [3.0, 4.0], Col.BASE_VOLUME: 100.0,
            Col.ELASTICITY: [-2.5, -0.8], Col.UNIT_COST: [1.8, 2.4], Col.PRICE_RATIO: [-0.05, 0.1],
            Col.ROBUST_P05: [2.7, 4.2], Col.ROBUST_P95: [3.0, 4.8],
            Col.EFFECTIVE_COST: [1.8, 2.4], Col.OWN_OPTIMAL_PRICE: [3.0, 4.8],
            Col.OBJECTIVE: "profit",
        }
    )  # fmt: skip
    for scenario, ratio in ((Scenario.CURRENT, 1.0), (Scenario.RECOMMENDED, 1.05)):
        frame[kpi_col(scenario, Metric.PRICE)] = frame[Col.CURRENT_PRICE] * ratio
        frame[kpi_col(scenario, Metric.PROFIT)] = 100.0
    return frame


class TestOptimizationPlotter:
    """All three charts render."""

    def test_price_changes(self) -> None:
        """One visible panel per category."""
        fig = OptimizationPlotter(PlotStyle()).price_changes(_recs(), (0.8, 1.2))
        assert isinstance(fig, Figure) and sum(ax.get_visible() for ax in fig.axes) == 2

    def test_objective_curve(self) -> None:
        """Renders for an elastic and an inelastic product."""
        plotter = OptimizationPlotter(PlotStyle())
        for upc in (1, 2):
            assert isinstance(plotter.objective_curve(_recs(), upc, (0.8, 1.2)), Figure)

    def test_scenarios(self) -> None:
        """Renders the scenario comparison."""
        summary = pd.DataFrame(
            {
                Col.SCENARIO: ["a", "b"],
                "volume_change": [0.0, -0.01],
                "revenue_change": [0.05, 0.04],
                "profit_change": [0.1, 0.12],
            }  # fmt: skip
        )
        assert isinstance(OptimizationPlotter(PlotStyle()).scenarios(summary), Figure)
