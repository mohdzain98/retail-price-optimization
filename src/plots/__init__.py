"""Plotting layer: one visual style for every project chart.

style.py (PlotStyle, FigureStore), eda.py (Phase 1 EdaPlotter) and
elasticity.py (Phase 2 ElasticityPlotter). Import in one line:
from src.plots import EdaPlotter, ElasticityPlotter, FigureStore, PlotStyle.
"""

from src.plots.eda import EdaPlotter
from src.plots.elasticity import ElasticityPlotter
from src.plots.style import FigureStore, PlotStyle

__all__ = ["EdaPlotter", "ElasticityPlotter", "FigureStore", "PlotStyle"]
