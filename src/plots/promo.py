"""Phase 5 charts: forecast backtest, budget frontier, promo calendar, scenarios, forecast.

Tactic colours follow the Tactic enum order (fixed slots). The frontier pairs the
gain curve with its marginal return in a second panel (no dual axis); the calendar
is a product x week heatmap of the MILP plan with the discount depth in each cell.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, PercentFormatter

from src.plots.style import PlotStyle
from src.schema import Col, PromoScenario, Tactic

TOTAL_METRICS: tuple[Col, ...] = (Col.VOLUME, Col.REVENUE, Col.PROFIT)
FORECAST_LINES: dict[Col, str] = {
    Col.FORECAST_CURRENT: "baseline, current prices",
    Col.FORECAST_BASELINE: "baseline, recommended prices",
    Col.PLAN_UNITS: "with MILP promo plan",
}
DOLLARS = FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k")


class PromoPlotter:
    """Draws the Phase 5 promo-planning charts, one objective at a time.

    Every method takes the long tables written by the promo stage and filters them.
    """

    def __init__(self, style: PlotStyle) -> None:
        self._style = style
        self._tactic_colors = style.ordered_colors([t.value for t in Tactic])

    def backtest(self, scores: pd.DataFrame) -> Figure:
        """Holdout WAPE per forecast method; the method used for the horizon is solid."""
        data = scores.sort_values(Col.WAPE, ascending=False)
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(data) + 1.4), layout="constrained")
        y = np.arange(len(data))
        alpha = np.where(data[Col.SELECTED], 1.0, 0.45)
        for i, (wape, a) in enumerate(zip(data[Col.WAPE], alpha, strict=True)):
            ax.barh(y[i], wape, 0.6, color=self._style.series[0], alpha=a)
        for i, (wape, bias) in enumerate(zip(data[Col.WAPE], data[Col.BIAS], strict=True)):
            ax.annotate(f"{wape:.1%}  (bias {bias:+.1%})", (wape, y[i]), xytext=(4, 0),
                        textcoords="offset points", va="center", color=self._style.ink)  # fmt: skip
        labels = [m.replace("_", " ") + ("  (used)" if sel else "")
                  for m, sel in zip(data[Col.METHOD], data[Col.SELECTED], strict=True)]  # fmt: skip
        ax.set_yticks(y, labels)
        ax.set_xlim(0, data[Col.WAPE].max() * 1.35)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("holdout WAPE of the weekly baseline (13 weeks, lower is better)")
        ax.set_title("Baseline forecast backtest", loc="left")
        ax.grid(axis="y", visible=False)
        return fig

    def frontier(self, frontier: pd.DataFrame, objective: str, available: float,
                 hurdle: float) -> Figure:  # fmt: skip
        """Best gain vs trade budget (top) and the marginal return of each step (bottom)."""
        data = frontier[frontier[Col.OBJECTIVE] == objective]
        fig, (top, bottom) = plt.subplots(
            2, 1, figsize=(9, 6), sharex=True, height_ratios=(2, 1), layout="constrained"
        )
        top.plot(data[Col.BUDGET], data[Col.GAIN], color=self._style.series[0], marker="o",
                 markersize=5, label=f"best incremental {objective} (MILP)")  # fmt: skip
        best = data[data[Col.RECOMMENDED_BUDGET]].iloc[0]
        top.scatter([best[Col.BUDGET]], [best[Col.GAIN]], s=90, color=self._style.ink, zorder=3,
                    marker="D", label=f"recommended ${best[Col.BUDGET]:,.0f}")  # fmt: skip
        for ax in (top, bottom):
            ax.axvline(available, color=self._style.ink_secondary, linestyle="--", linewidth=0.8)
        top.annotate("available budget", (available, 0), xytext=(4, 4),
                     textcoords="offset points", color=self._style.ink_secondary)  # fmt: skip
        top.set_ylabel(f"incremental {objective} over the horizon")
        top.yaxis.set_major_formatter(DOLLARS)
        top.set_title(f"Promo budget frontier (goal: {objective})", loc="left")
        top.legend(loc="upper left")
        steps = data[Col.BUDGET].diff()
        bottom.bar(data[Col.BUDGET], data[Col.MARGINAL_RETURN], width=-0.9 * steps,
                   align="edge", color=self._style.series[0], alpha=0.6)  # fmt: skip
        bottom.axhline(hurdle, color=self._style.series[1], linewidth=1.2,
                       label=f"hurdle {hurdle:.2f} per $1")  # fmt: skip
        bottom.set_ylabel("marginal return per $1")
        bottom.set_xlabel("trade budget (discount funding + fixed fees)")
        bottom.xaxis.set_major_formatter(DOLLARS)
        bottom.legend(loc="upper right")
        return fig

    def calendar(self, forecast: pd.DataFrame, objective: str) -> Figure:
        """MILP promo calendar: promoted products x weeks, coloured by tactic, depth labelled."""
        data = forecast[forecast[Col.OBJECTIVE] == objective]
        promoted = data.groupby(Col.UPC)[Col.TACTIC].count()
        data = data[data[Col.UPC].isin(promoted[promoted > 0].index)]
        label = data[Col.CATEGORY].str.title() + " | " + data[Col.DESCRIPTION].str[:22] + " " \
            + data[Col.PRODUCT_SIZE]  # fmt: skip
        data = data.assign(label=label).sort_values([Col.CATEGORY, Col.DESCRIPTION])
        tactics = [t.value for t in Tactic]
        code = data[Col.TACTIC].map({t: i for i, t in enumerate(tactics)}).fillna(-1)
        grid = data.assign(code=code).pivot_table(
            index="label", columns=Col.HORIZON_WEEK, values="code", sort=False
        )
        depth = data.pivot_table(index="label", columns=Col.HORIZON_WEEK,
                                 values=Col.DISCOUNT_DEPTH, sort=False)  # fmt: skip
        weeks = data.drop_duplicates(Col.HORIZON_WEEK).set_index(Col.HORIZON_WEEK)[Col.WEEK]
        fig, ax = plt.subplots(figsize=(11, 0.28 * len(grid) + 1.8), layout="constrained")
        cmap = ListedColormap(["#ffffff", *(self._tactic_colors[t] for t in tactics)])
        ax.imshow(grid.to_numpy() + 1, cmap=cmap, vmin=0, vmax=len(tactics), aspect="auto")
        for (i, j), value in np.ndenumerate(depth.reindex_like(grid).to_numpy()):
            if not np.isnan(value):
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=7,
                        color=self._style.ink)  # fmt: skip
        ax.set_xticks(np.arange(grid.shape[1]), [weeks[w].strftime("%b %d") for w in grid.columns],
                      rotation=45, ha="right")  # fmt: skip
        ax.set_yticks(np.arange(len(grid)), grid.index, fontsize=7)
        ax.set_xticks(np.arange(-0.5, grid.shape[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, len(grid)), minor=True)
        ax.grid(which="minor", color="#ffffff", linewidth=2)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)
        handles = [
            Patch(color=self._tactic_colors[t], label=t.replace("_", " + ")) for t in tactics
        ]
        fig.legend(handles=handles, ncols=len(tactics), loc="outside lower center")
        n_all = forecast.loc[forecast[Col.OBJECTIVE] == objective, Col.UPC].nunique()
        ax.set_title(f"MILP promo calendar (goal: {objective}): {len(grid)} of {n_all} products "
                     "promoted, cell = discount depth", loc="left")  # fmt: skip
        return fig

    def scenarios(self, summary: pd.DataFrame, objective: str) -> Figure:
        """Volume, revenue and profit change vs current prices with no promo, per scenario."""
        data = summary[summary[Col.OBJECTIVE] == objective]
        order = [s.value for s in PromoScenario]
        data = data.set_index(Col.SCENARIO).loc[[s for s in order if s in set(data[Col.SCENARIO])]]
        fig, ax = plt.subplots(figsize=(9, 0.75 * len(data) + 1.5), layout="constrained")
        y, height = np.arange(len(data)), 0.26
        for i, metric in enumerate(TOTAL_METRICS):
            ax.barh(y + (i - 1) * height, data[f"{metric.value}_change"], height - 0.03,
                    color=self._style.series[i], label=metric.value)  # fmt: skip
        labels = [f"{s.replace('_', ' ')}\nspend ${spend:,.0f}, {n} promos"
                  for s, spend, n in zip(data.index, data[Col.TRADE_SPEND], data[Col.N_PROMOS],
                                         strict=True)]  # fmt: skip
        ax.set_yticks(y, labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color=self._style.ink_secondary, linewidth=0.8)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("change vs current prices with no promotions (forecast horizon)")
        ax.set_title(f"Horizon KPIs by scenario (goal: {objective})", loc="left")
        ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        return fig

    def forecast(self, forecast: pd.DataFrame, objective: str) -> Figure:
        """Weekly portfolio units: baseline at current and recommended prices, and with promos."""
        data = forecast[forecast[Col.OBJECTIVE] == objective]
        weekly = data.groupby(Col.WEEK)[list(FORECAST_LINES)].sum()
        fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
        for i, (col, label) in enumerate(FORECAST_LINES.items()):
            ax.plot(weekly.index, weekly[col], color=self._style.series[i], label=label,
                    marker="o" if col == Col.PLAN_UNITS else None, markersize=4)  # fmt: skip
        ax.set_ylim(bottom=0)
        ax.set_ylabel("units per week (all products)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
        ax.set_title(f"Forecast horizon (goal: {objective})", loc="left")
        ax.legend(loc="lower left")
        fig.autofmt_xdate()
        return fig
