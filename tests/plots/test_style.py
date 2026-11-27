"""Tests for src.plots.style: colour assignment and figure saving.

Colour must follow the entity, never its rank, and FigureStore must write a
PNG to the folder it was given.
One test class per production class.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from src.plots import FigureStore, PlotStyle

matplotlib.use("Agg")


class TestPlotStyle:
    """PlotStyle assigns colours by entity, not by rank."""

    def test_category_colors_stable(self) -> None:
        """Same categories map to the same colours regardless of input order."""
        style = PlotStyle()
        assert style.category_colors(["b", "a"]) == style.category_colors(["a", "b"])
        assert style.ordered_colors(["x", "y"])["y"] == style.series[1]


class TestFigureStore:
    """FigureStore writes PNG files."""

    def test_save(self, tmp_path: Path) -> None:
        """Saving creates <dir>/<name>.png."""
        fig, _ = plt.subplots()
        path = FigureStore(tmp_path / "figs").save(fig, "demo")
        assert path == tmp_path / "figs" / "demo.png" and path.exists()
