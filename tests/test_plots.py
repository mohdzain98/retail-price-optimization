"""Smoke tests for src.plots: every EDA chart renders and saves.

Uses the synthetic end-to-end panel and the non-interactive Agg backend.
Visual quality is checked by eye; these tests guard against crashes.
One test class per production class.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.plots import EdaPlotter, PlotStyle

matplotlib.use("Agg")


class TestPlotStyle:
    """PlotStyle assigns colours by entity, not by rank."""

    def test_category_colors_stable(self) -> None:
        """Same categories map to the same colours regardless of input order."""
        style = PlotStyle()
        assert style.category_colors(["b", "a"]) == style.category_colors(["a", "b"])
        assert style.category_colors(["a"])["a"] == style.series[0]


class TestEdaPlotter:
    """EdaPlotter charts render and save as PNG."""

    @pytest.mark.parametrize("chart", ["coverage_heatmap", "price_vs_units", "promo_frequency"])
    def test_panel_charts(self, chart: str, panel: pd.DataFrame, tmp_path: Path) -> None:
        """Each panel-level chart returns a Figure that saves to disk."""
        plotter = EdaPlotter(PlotStyle(), tmp_path)
        fig = getattr(plotter, chart)(panel)
        assert isinstance(fig, Figure)
        assert plotter.save(fig, chart).exists()

    def test_trend_overlay(self, panel: pd.DataFrame, tmp_path: Path) -> None:
        """Trend small multiples render for two products."""
        fig = EdaPlotter(PlotStyle(), tmp_path).trend_overlay(panel, [1111, 3333])
        assert len(fig.axes) == 2
