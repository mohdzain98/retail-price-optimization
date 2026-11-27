"""Smoke tests for src.plots.elasticity.ElasticityPlotter.

Builds a small all-models table and comparison table by hand so the charts can
render without fitting models. One test class per production class.
"""

from __future__ import annotations

import matplotlib
import pandas as pd
from matplotlib.figure import Figure

from src.config import ModelName
from src.plots import ElasticityPlotter, PlotStyle
from src.schema import Col

matplotlib.use("Agg")


def _all_models() -> pd.DataFrame:
    """Two models x three products in two categories."""
    rows = [
        {
            Col.UPC: u,
            Col.DESCRIPTION: f"ITEM {u}",
            Col.PRODUCT_SIZE: "1 OZ",
            Col.CATEGORY: c,
            Col.MODEL: m.value,
            Col.ELASTICITY: b,
            Col.ELASTICITY_LOW: b - 0.3,
            Col.ELASTICITY_HIGH: b + 0.3,
        }
        for m in (ModelName.OLS_PER_PRODUCT, ModelName.BAYES_HIERARCHICAL)
        for u, c, b in ((1, "A", -1.2), (2, "A", -2.0), (3, "B", -0.9))
    ]
    return pd.DataFrame(rows)


class TestElasticityPlotter:
    """Forest and comparison charts render."""

    def test_forest(self) -> None:
        """One visible panel per category."""
        data = _all_models()
        final = data[data[Col.MODEL] == ModelName.BAYES_HIERARCHICAL.value]
        fig = ElasticityPlotter(PlotStyle()).forest(data, final, (-5.0, -0.3))
        assert isinstance(fig, Figure)
        assert sum(ax.get_visible() for ax in fig.axes) == 2

    def test_comparison(self) -> None:
        """Bar chart renders for the models present."""
        cmp = pd.DataFrame(
            {Col.MODEL: [m.value for m in ModelName], Col.WAPE: [0.3, 0.25, 0.2, 0.18]}
        )
        assert isinstance(ElasticityPlotter(PlotStyle()).comparison(cmp), Figure)
