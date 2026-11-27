"""Tests for src.simulator.charts: every Altair chart builds a valid Vega-Lite spec.

Charts are built from real simulator outputs on the synthetic data and serialised
with to_dict(), which validates the spec. One test class per production class.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import Config, Objective
from src.plots.style import PlotStyle
from src.schema import Col
from src.simulator import (
    PriceSettings,
    PriceSimulator,
    PromoSettings,
    PromoSimulator,
    SimulatorCharts,
    SimulatorData,
)


@pytest.fixture(scope="module")
def data(processed_dir: Path) -> SimulatorData:
    """Synthetic processed outputs."""
    return SimulatorData.load(processed_dir)


class TestSimulatorCharts:
    """All charts serialise to valid specs."""

    def test_price_charts(self, data: SimulatorData) -> None:
        """Price changes and objective curve."""
        sim = PriceSimulator(Config.load(), data.recommendations)
        settings = PriceSettings.from_config(Config.load(), sim.categories)
        charts = SimulatorCharts(PlotStyle(), sim.categories)
        result = sim.optimize(Objective.PROFIT, settings)
        curve = sim.curve(Objective.PROFIT, settings, int(sim.products[Col.UPC].iloc[0]))
        marks = pd.DataFrame({"label": ["Current"], "price": [1.0], "value": [1.0]})
        for chart in (charts.price_changes(result.products),
                      charts.objective_curve(curve, marks, "profit")):  # fmt: skip
            assert chart.to_dict()

    def test_promo_and_forecast_charts(self, data: SimulatorData) -> None:
        """Scenarios, calendar, weekly, frontier and backtest."""
        sim = PromoSimulator(data)
        settings = PromoSettings.from_config(Config.load())
        result = sim.plan("revenue", settings)
        charts = SimulatorCharts(PlotStyle(), sorted(data.forecast[Col.CATEGORY].unique()))
        frontier = sim.frontier("revenue", settings, 3, 0.25)
        built = (
            charts.scenarios(result.summary), charts.calendar(result.weekly),
            charts.weekly(result.weekly), charts.frontier(frontier, settings.budget, "revenue"),
            charts.backtest(data.backtest),
        )  # fmt: skip
        assert all(chart.to_dict() for chart in built)
