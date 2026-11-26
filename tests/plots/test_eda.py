"""Smoke tests for src.plots.eda.EdaPlotter on the synthetic end-to-end panel.

Visual quality is checked by eye; these tests guard against crashes and make
sure each chart returns a Figure with the expected number of panels.
One test class per production class.
"""

from __future__ import annotations

import matplotlib
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.plots import EdaPlotter, PlotStyle

matplotlib.use("Agg")


class TestEdaPlotter:
    """EdaPlotter charts render."""

    @pytest.mark.parametrize("chart", ["coverage_heatmap", "price_vs_units", "promo_frequency"])
    def test_panel_charts(self, chart: str, panel: pd.DataFrame) -> None:
        """Each panel-level chart returns a Figure."""
        assert isinstance(getattr(EdaPlotter(PlotStyle()), chart)(panel), Figure)

    def test_trend_overlay(self, panel: pd.DataFrame) -> None:
        """Trend small multiples render one axis per product."""
        fig = EdaPlotter(PlotStyle()).trend_overlay(panel, [1111, 3333])
        assert len(fig.axes) == 2
