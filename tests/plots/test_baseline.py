"""Smoke tests for src.plots.baseline.BaselinePlotter.

Builds a tiny decomposed weekly table by hand so both charts render, including a
negative pantry lift. One test class per production class.
"""

from __future__ import annotations

import matplotlib
import pandas as pd
from matplotlib.figure import Figure

from src.plots import BaselinePlotter, PlotStyle
from src.schema import LIFT_COLS, Col, EffectTerm

matplotlib.use("Agg")


def _weekly() -> pd.DataFrame:
    """Two products x 4 weeks with a promo spike and a pantry dip."""
    rows = []
    for upc, cat in ((1, "A"), (2, "B")):
        for i, week in enumerate(pd.date_range("2010-01-02", periods=4, freq="7D")):
            promo = i == 1
            rows.append(
                {
                    Col.UPC: upc, Col.CATEGORY: cat, Col.DESCRIPTION: f"ITEM {upc}",
                    Col.PRODUCT_SIZE: "1 OZ", Col.WEEK: week, Col.BASELINE: 100.0,
                    Col.UNITS: 160.0 if promo else 95.0 if i == 2 else 100.0,
                    LIFT_COLS[EffectTerm.PRICE]: 40.0 if promo else 0.0,
                    LIFT_COLS[EffectTerm.PROMO]: 20.0 if promo else 0.0,
                    LIFT_COLS[EffectTerm.PANTRY]: -5.0 if i == 2 else 0.0,
                    LIFT_COLS[EffectTerm.ERROR]: 0.0,
                }
            )  # fmt: skip
    return pd.DataFrame(rows)


class TestBaselinePlotter:
    """Both baseline charts render."""

    def test_weekly(self) -> None:
        """One axis per product."""
        fig = BaselinePlotter(PlotStyle()).weekly(_weekly(), [1, 2])
        assert isinstance(fig, Figure) and len(fig.axes) == 2

    def test_category_due_to(self) -> None:
        """Renders with a negative (pantry) layer."""
        assert isinstance(BaselinePlotter(PlotStyle()).category_due_to(_weekly()), Figure)
