"""Phase 1 EDA charts over the modelling panel.

EdaPlotter draws coverage, price-vs-units, promo frequency and trend charts.
Uses PlotStyle for all colours; FigureStore saves the returned figures.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from src.plots.style import PlotStyle
from src.schema import PROMO_COLS, Col, Split


class EdaPlotter:
    """Phase 1 EDA charts over the panel; every method returns a Figure.

    Saving is left to FigureStore.
    """

    def __init__(self, style: PlotStyle) -> None:
        self._style = style

    def coverage_heatmap(self, panel: pd.DataFrame) -> Figure:
        """Weekly log units per product, rows grouped by category; blank = no sales."""
        order = panel.drop_duplicates(Col.UPC).sort_values([Col.CATEGORY, Col.UPC])[Col.UPC]
        grid = panel.pivot_table(index=Col.UPC, columns=Col.WEEK, values=Col.UNITS, aggfunc="sum")
        grid = np.log1p(grid.reindex(order))
        fig, ax = plt.subplots(figsize=(10, 0.14 * len(grid) + 1.5))
        image = ax.imshow(grid.to_numpy(), aspect="auto", cmap=self._style.sequential_cmap)
        ticks = np.linspace(0, grid.shape[1] - 1, 6).astype(int)
        ax.set_xticks(ticks, [grid.columns[i].strftime("%Y-%m") for i in ticks])
        ax.set_yticks([])
        ax.set_ylabel(f"{len(grid)} products (grouped by category)")
        ax.set_title("Weekly units per product (log scale); blank = no sales")
        ax.grid(False)
        fig.colorbar(image, ax=ax, label="log(1 + units)", fraction=0.025)
        return fig

    def price_vs_units(self, panel: pd.DataFrame) -> Figure:
        """Within-product (demeaned) log price vs log units for eligible products.

        One panel per category; the OLS slope is a naive elasticity (no controls).
        """
        eligible = panel[panel[Col.ELIGIBLE]]
        by_upc = eligible.groupby(Col.UPC)
        demeaned = eligible.assign(
            x=eligible[Col.LN_PRICE] - by_upc[Col.LN_PRICE].transform("mean"),
            y=eligible[Col.LN_UNITS] - by_upc[Col.LN_UNITS].transform("mean"),
        )
        cats = sorted(demeaned[Col.CATEGORY].dropna().unique())
        colors = self._style.category_colors(cats)
        fig, axes = plt.subplots(
            1, len(cats), figsize=(3.2 * len(cats), 3.4), sharey=True, layout="constrained"
        )
        for ax, cat in zip(np.atleast_1d(axes), cats, strict=True):
            self._scatter_with_fit(ax, demeaned[demeaned[Col.CATEGORY] == cat], colors[cat], cat)
        np.atleast_1d(axes)[0].set_ylabel("log units (demeaned)")
        fig.suptitle("Price vs units within eligible products (naive slope, no controls)")
        return fig

    def promo_frequency(self, panel: pd.DataFrame) -> Figure:
        """Share of product-weeks where any store runs each tactic, by category."""
        promoted = panel.assign(**{c: panel[c] > 0 for c in PROMO_COLS})
        share = promoted.groupby(Col.CATEGORY)[list(PROMO_COLS)].mean()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        width, x = 0.26, np.arange(len(share))
        for i, tactic in enumerate(PROMO_COLS):
            offset = x + (i - 1) * width
            ax.bar(offset, share[tactic], width - 0.03, color=self._style.series[i], label=tactic)
        ax.set_xticks(x, [c.title() for c in share.index])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylabel("share of product-weeks")
        ax.set_title("Promo frequency by category")
        ax.legend(ncols=len(PROMO_COLS), loc="upper right")
        return fig

    def trend_overlay(self, panel: pd.DataFrame, upcs: list[int]) -> Figure:
        """Log units with LOESS trend per product (small multiples); holdout shaded."""
        fig, axes = plt.subplots(
            len(upcs), 1, figsize=(9, 2.1 * len(upcs) + 0.4), sharex=True, layout="constrained"
        )
        for ax, upc in zip(np.atleast_1d(axes), upcs, strict=True):
            self._trend_panel(ax, panel[panel[Col.UPC] == upc].sort_values(Col.WEEK), upc)
        handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncols=3, loc="outside upper right")
        return fig

    def _scatter_with_fit(self, ax: Axes, data: pd.DataFrame, color: str, category: str) -> None:
        """Scatter x/y with an OLS line and slope label on one axis."""
        ax.scatter(data["x"], data["y"], s=8, alpha=0.35, color=color, linewidths=0)
        slope, intercept = np.polyfit(data["x"], data["y"], 1)
        xs = np.linspace(data["x"].min(), data["x"].max(), 50)
        ax.plot(xs, intercept + slope * xs, color=self._style.ink, linewidth=1.5)
        ax.set_title(f"{category.title()} ({data[Col.UPC].nunique()} products)")
        ax.text(0.03, 0.05, f"slope ≈ {slope:.2f}", transform=ax.transAxes, color=self._style.ink)
        ax.set_xlabel("log price (demeaned)")

    def _trend_panel(self, ax: Axes, series: pd.DataFrame, upc: int) -> None:
        """One product's log units, trend line and shaded holdout."""
        weeks, style = series[Col.WEEK], self._style
        ax.plot(weeks, series[Col.LN_UNITS], color=style.series[0], linewidth=1, label="log units")
        ax.plot(weeks, series[Col.LN_TREND], color=style.series[1], label="LOESS trend")
        test_weeks = weeks[series[Col.SPLIT] == Split.TEST]
        if len(test_weeks):
            ax.axvspan(
                test_weeks.min(), test_weeks.max(), color=style.grid, alpha=0.6, label="holdout"
            )
        ax.set_title(f"{series[Col.DESCRIPTION].iloc[0]} ({upc})", loc="left")
