"""Phase 2 charts: per-product elasticity by model and holdout accuracy.

The forest plot shows each model's regular or promo elasticity and interval per
product, one panel per category, so shrinkage toward the category is visible.
The comparison chart ranks models by holdout WAPE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.config import ModelName
from src.plots.style import PlotStyle
from src.schema import ESTIMATE_COLS, Col, Family

LABEL_CHARS = 24
ROW_HEIGHT = 0.3
DODGE = 0.17


@dataclass(frozen=True)
class _ForestSpec:
    """What a forest panel draws: models in ladder order, bounds and family."""

    models: list[str]
    bounds: tuple[float, float]
    family: Family


class ElasticityPlotter:
    """Draws elasticity forest plots and the model comparison bar chart.

    Model colours follow ModelName order, so a model keeps its colour everywhere.
    """

    def __init__(self, style: PlotStyle) -> None:
        self._style = style
        self._colors = style.ordered_colors([m.value for m in ModelName])

    def forest(
        self,
        all_models: pd.DataFrame,
        final: pd.DataFrame,
        bounds: tuple[float, float],
        family: Family = Family.REGULAR,
    ) -> Figure:
        """Estimates with intervals per product and model, one panel per category.

        Products are ordered by the final elasticity; dashed lines mark the bounds.
        """
        cats = sorted(final[Col.CATEGORY].unique())
        models = [m.value for m in ModelName if m.value in set(all_models[Col.MODEL])]
        spec = _ForestSpec(models=models, bounds=bounds, family=family)
        rows = int(final.groupby(Col.CATEGORY).size().max())
        n_rows = math.ceil(len(cats) / 2)
        fig, axes = plt.subplots(
            n_rows, 2, figsize=(13, n_rows * (rows * ROW_HEIGHT + 1.2)),
            sharex=True, layout="constrained",
        )  # fmt: skip
        estimate = ESTIMATE_COLS[family][0]
        for ax, cat in zip(axes.flat, cats, strict=False):
            order = final[final[Col.CATEGORY] == cat].sort_values(estimate)
            self._forest_panel(ax, all_models[all_models[Col.CATEGORY] == cat], order, spec)
            ax.set_title(cat.title(), loc="left")
        for ax in list(axes.flat)[len(cats) :]:
            ax.set_visible(False)
        handles, labels = list(axes.flat)[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncols=len(models), loc="outside upper center")
        fig.supxlabel(f"{family.value} price elasticity (dot = estimate, line = interval)")
        return fig

    def comparison(self, comparison: pd.DataFrame) -> Figure:
        """Holdout WAPE per model (lower is better), in model-ladder order."""
        order = [m.value for m in ModelName if m.value in set(comparison[Col.MODEL])]
        data = comparison.set_index(Col.MODEL).loc[order]
        fig, ax = plt.subplots(figsize=(7, 0.55 * len(order) + 1.2))
        y = np.arange(len(order))
        ax.barh(y, data[Col.WAPE], height=0.6, color=[self._colors[m] for m in order])
        for yi, value in zip(y, data[Col.WAPE], strict=True):
            ax.text(float(value), float(yi), f" {value:.1%}", va="center", color=self._style.ink)
        ax.set_yticks(y, order)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlabel("holdout WAPE on units (lower is better)")
        ax.set_title("Model accuracy on the 13-week holdout")
        return fig

    def _forest_panel(
        self, ax: Axes, estimates: pd.DataFrame, order: pd.DataFrame, spec: _ForestSpec
    ) -> None:
        """One category: dodge each model's dot and interval around the product row."""
        est, low, high = ESTIMATE_COLS[spec.family]
        y = np.arange(len(order))
        for i, model in enumerate(spec.models):
            by_upc = estimates[estimates[Col.MODEL] == model].set_index(Col.UPC)
            rows = by_upc.reindex(order[Col.UPC])
            offset = y + (i - (len(spec.models) - 1) / 2) * DODGE
            color = self._colors[model]
            ax.hlines(offset, rows[low], rows[high], color=color, linewidth=1)
            ax.scatter(rows[est], offset, s=12, color=color, label=model, zorder=3)
        bounds = spec.bounds
        for bound in bounds:
            ax.axvline(bound, color=self._style.ink_secondary, linestyle="--", linewidth=0.8)
        labels = [
            f"{str(d)[:LABEL_CHARS]} {s}"
            for d, s in zip(order[Col.DESCRIPTION], order[Col.PRODUCT_SIZE], strict=True)
        ]  # sizes tell apart same-name packs
        ax.set_yticks(y, labels, fontsize=7)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
