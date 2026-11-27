"""Phase 3 charts: baseline vs actual volume and the due-to decomposition.

The weekly chart stacks baseline and the positive lifts under the actual-units
line for one product; the category chart splits each category's volume into
baseline and drivers, with negative drivers (pantry dips) drawn left of zero.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from src.plots.style import PlotStyle
from src.schema import LIFT_COLS, Col, EffectTerm

LAYERS: tuple[str, ...] = (Col.BASELINE, *LIFT_COLS.values())


class BaselinePlotter:
    """Draws the weekly decomposition and the category due-to chart.

    Layer colours follow a fixed order (baseline, price, promo, pantry, error).
    """

    def __init__(self, style: PlotStyle) -> None:
        self._style = style
        self._colors = style.ordered_colors(list(LAYERS))
        self._labels = {
            Col.BASELINE: "baseline",
            **{LIFT_COLS[t]: f"{t.value} lift" for t in EffectTerm},
        }

    def weekly(self, weekly: pd.DataFrame, upcs: list[int]) -> Figure:
        """Small multiples: stacked baseline + positive lifts, with actual units as a line."""
        fig, axes = plt.subplots(
            len(upcs), 1, figsize=(10, 2.4 * len(upcs) + 0.5), sharex=True, layout="constrained"
        )
        for ax, upc in zip(np.atleast_1d(axes), upcs, strict=True):
            self._weekly_panel(ax, weekly[weekly[Col.UPC] == upc].sort_values(Col.WEEK))
        handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncols=len(labels), loc="outside upper center")
        return fig

    def category_due_to(self, weekly: pd.DataFrame) -> Figure:
        """Share of each category's units by baseline and driver (negatives left of 0)."""
        totals = weekly.groupby(Col.CATEGORY)[[Col.UNITS, *LAYERS]].sum()
        shares = totals[list(LAYERS)].div(totals[Col.UNITS], axis=0)
        fig, ax = plt.subplots(figsize=(9, 0.6 * len(shares) + 1.6), layout="constrained")
        y = np.arange(len(shares))
        right, left = np.zeros(len(shares)), np.zeros(len(shares))
        for layer in LAYERS:
            values = shares[layer].to_numpy()
            start = np.where(values >= 0, right, left)
            ax.barh(
                y,
                values,
                left=start,
                height=0.6,
                color=self._colors[layer],
                label=self._labels[layer],
            )
            right, left = right + values.clip(min=0), left + values.clip(max=0)
        ax.set_yticks(y, [c.title() for c in shares.index])
        ax.invert_yaxis()
        ax.axvline(0, color=self._style.ink_secondary, linewidth=0.8)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("share of category units")
        ax.set_title("What drives volume: baseline vs incremental drivers", loc="left")
        ax.legend(ncols=len(LAYERS), loc="upper center", bbox_to_anchor=(0.5, -0.25))
        return fig

    def _weekly_panel(self, ax: Axes, series: pd.DataFrame) -> None:
        """One product: stacked baseline and positive lifts, actual units line."""
        weeks = series[Col.WEEK]
        layers = [series[c].clip(lower=0) for c in LAYERS]
        colors = [self._colors[c] for c in LAYERS]
        labels = [self._labels[c] for c in LAYERS]
        ax.stackplot(weeks, *layers, colors=colors, labels=labels, alpha=0.85, linewidth=0)
        ax.plot(weeks, series[Col.UNITS], color=self._style.ink, linewidth=1, label="actual units")
        name = f"{series[Col.DESCRIPTION].iloc[0]} {series[Col.PRODUCT_SIZE].iloc[0]}"
        ax.set_title(name, loc="left")
        ax.set_ylabel("units")
