"""Plotting layer: one visual style for every project chart.

style.py (PlotStyle, FigureStore), eda.py (Phase 1), elasticity.py (Phase 2)
and baseline.py (Phase 3). Import in one line:
from src.plots import BaselinePlotter, FigureStore, PlotStyle.
"""

from src.plots.baseline import BaselinePlotter
from src.plots.eda import EdaPlotter
from src.plots.elasticity import ElasticityPlotter
from src.plots.style import FigureStore, PlotStyle

__all__ = ["BaselinePlotter", "EdaPlotter", "ElasticityPlotter", "FigureStore", "PlotStyle"]
