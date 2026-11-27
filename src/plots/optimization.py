"""Phase 4 charts: recommended price changes, a profit-curve explainer and scenarios.

Price changes show each product's recommended move with its robustness range
(5th-95th percentile across elasticity draws). The objective curve contrasts the
constant-elasticity and linear demand curves for one product and one objective.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from src.plots.style import PlotStyle
from src.schema import Col, Metric, Scenario, kpi_col

CURVE_POINTS = 200
SUMMARY_METRICS: tuple[Metric, ...] = (Metric.VOLUME, Metric.REVENUE, Metric.PROFIT)


class OptimizationPlotter:
    """Draws the Phase 4 recommendation charts.

    Colours: categories by sorted name; scenarios and metrics in fixed order.
    """

    def __init__(self, style: PlotStyle) -> None:
        self._style = style

    def price_changes(self, recs: pd.DataFrame, bounds: tuple[float, float]) -> Figure:
        """Recommended % price change per product with robustness range, one panel per category."""
        cats = sorted(recs[Col.CATEGORY].unique())
        colors = self._style.category_colors(cats)
        rows = int(recs.groupby(Col.CATEGORY).size().max())
        n_rows = math.ceil(len(cats) / 2)
        fig, axes = plt.subplots(
            n_rows, 2, figsize=(13, n_rows * (0.3 * rows + 1.2)), sharex=True, layout="constrained"
        )
        for ax, cat in zip(axes.flat, cats, strict=False):
            self._change_panel(ax, recs[recs[Col.CATEGORY] == cat], colors[cat], bounds)
            ax.set_title(cat.title(), loc="left")
        for ax in list(axes.flat)[len(cats) :]:
            ax.set_visible(False)
        fig.supxlabel(
            "recommended regular-price change (line = 5th-95th pct across elasticity draws)"
        )
        return fig

    def objective_curve(self, recs: pd.DataFrame, upc: int, bounds: tuple[float, float]) -> Figure:
        """Objective value vs price for one product (one objective's rows) under both curves.

        Plots sum((P - c_eff) Q): profit for the profit objective, revenue for revenue.
        Marks the current price, the product's own optimum and the portfolio recommendation.
        """
        row = recs.set_index(Col.UPC).loc[upc]
        p0, base, beta = (row[c] for c in (Col.CURRENT_PRICE, Col.BASE_VOLUME, Col.ELASTICITY))
        c_eff, objective = row[Col.EFFECTIVE_COST], row[Col.OBJECTIVE]
        prices = np.linspace(0.7 * p0, 1.3 * p0, CURVE_POINTS)

        def value(price: np.ndarray) -> np.ndarray:
            return (price - c_eff) * base * (price / p0) ** beta

        linear = (prices - c_eff) * np.maximum(base * (1 + beta * (prices / p0 - 1)), 0)
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        low, high = bounds[0] * p0, bounds[1] * p0
        ax.axvspan(low, high, color=self._style.grid, alpha=0.5, label="allowed range")
        ax.plot(
            prices, value(prices), color=self._style.series[0], label="constant elasticity (used)"
        )
        ax.plot(prices, linear, color=self._style.series[1], linestyle="--", label="linear formula")
        points = (
            ("current", row[Col.CURRENT_PRICE], "o", self._style.ink),
            ("own optimum", row[Col.OWN_OPTIMAL_PRICE], "^", self._style.series[0]),
            ("recommended", row[kpi_col(Scenario.RECOMMENDED, Metric.PRICE)], "D", self._style.ink),
        )
        for label, price, marker, color in points:
            ax.scatter([price], [value(np.array(price))], color=color, marker=marker, s=50,
                       zorder=3, label=f"{label} ${price:.2f}")  # fmt: skip
        ax.set_xlabel("regular price ($)")
        ax.set_ylabel(f"weekly {objective} ($)")
        title = (
            f"{row[Col.DESCRIPTION]} {row[Col.PRODUCT_SIZE]}  (β = {beta:.2f}, goal: {objective})"
        )
        ax.set_title(title, loc="left")
        ax.legend(loc="lower center", fontsize=8)
        return fig

    def scenarios(self, summary: pd.DataFrame) -> Figure:
        """Volume, revenue and profit change vs current for each scenario."""
        fig, ax = plt.subplots(figsize=(9, 0.7 * len(summary) + 1.5), layout="constrained")
        y = np.arange(len(summary))
        height = 0.26
        for i, metric in enumerate(SUMMARY_METRICS):
            values = summary[f"{metric.value}_change"].to_numpy()
            ax.barh(
                y + (i - 1) * height,
                values,
                height - 0.03,
                color=self._style.series[i],
                label=metric.value,
            )
        ax.set_yticks(y, summary[Col.SCENARIO])
        ax.invert_yaxis()
        ax.axvline(0, color=self._style.ink_secondary, linewidth=0.8)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("change vs current prices (weekly, baseline business)")
        ax.set_title("Portfolio impact by solver and scenario", loc="left")
        ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
        return fig

    def _change_panel(
        self, ax: plt.Axes, recs: pd.DataFrame, color: str, bounds: tuple[float, float]
    ) -> None:
        """Dot = recommended change, line = robustness range, dashed = bounds."""
        order = recs.sort_values(Col.PRICE_RATIO)
        y = np.arange(len(order))
        p0 = order[Col.CURRENT_PRICE]
        if Col.ROBUST_P05 in order and order[Col.ROBUST_P05].notna().any():
            ax.hlines(
                y,
                order[Col.ROBUST_P05] / p0 - 1,
                order[Col.ROBUST_P95] / p0 - 1,
                color=color,
                linewidth=1.2,
            )
        ax.scatter(order[Col.PRICE_RATIO], y, color=color, s=16, zorder=3)
        for bound in bounds:
            ax.axvline(bound - 1, color=self._style.ink_secondary, linestyle="--", linewidth=0.8)
        ax.axvline(0, color=self._style.ink_secondary, linewidth=0.8)
        labels = [
            f"{str(d)[:24]} {s}"
            for d, s in zip(order[Col.DESCRIPTION], order[Col.PRODUCT_SIZE], strict=True)
        ]
        ax.set_yticks(y, labels, fontsize=7)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", visible=False)
