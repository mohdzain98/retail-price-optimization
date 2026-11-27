"""Interactive Altair charts for the Streamlit simulator (hover tooltips on every mark).

SimulatorCharts mirrors the report figures: price changes, objective curve, scenario
KPIs, promo calendar, budget frontier, weekly forecast and forecast backtest.
Categorical colours come from PlotStyle in fixed order, so they match the figures.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from src.plots.style import PlotStyle
from src.schema import Col, Metric, PromoScenario, Scenario, Tactic, kpi_col

SCENARIO_LABELS: dict[str, str] = {
    PromoScenario.CURRENT_NO_PROMO.value: "Current prices, no promo",
    PromoScenario.RECOMMENDED_NO_PROMO.value: "Recommended prices, no promo",
    PromoScenario.LAST_YEAR_PLAN.value: "Last year's calendar",
    PromoScenario.MILP_SAME_SPEND.value: "MILP at last year's spend",
    PromoScenario.GREEDY_PLAN.value: "Greedy plan",
    PromoScenario.MILP_PLAN.value: "MILP plan",
    "your_plan": "Your MILP plan",
}
TACTIC_LABELS: dict[str, str] = {
    Tactic.FEATURE.value: "Feature",
    Tactic.DISPLAY.value: "Display",
    Tactic.TPR.value: "TPR",
    Tactic.FEATURE_DISPLAY.value: "Feature + display",
}
METRIC_LABELS: dict[str, str] = {"volume": "Volume", "revenue": "Revenue", "profit": "Profit"}
FORECAST_LABELS: dict[str, str] = {
    Col.FORECAST_CURRENT: "Baseline, current prices",
    Col.FORECAST_BASELINE: "Baseline, recommended prices",
    Col.PLAN_UNITS: "With promo plan",
}


class SimulatorCharts:
    """Builds the simulator's Altair charts from simulator outputs.

    Categories, tactics, metrics and series take fixed palette slots (never by rank).
    """

    def __init__(self, style: PlotStyle, categories: list[str]) -> None:
        self._style = style
        self._categories = sorted(categories)
        self._category_scale = alt.Scale(
            domain=self._categories, range=list(style.series[: len(self._categories)])
        )

    def price_changes(self, products: pd.DataFrame) -> alt.Chart:
        """Recommended % change per product, coloured by category, sorted by change."""
        data = products.assign(
            label=products[Col.DESCRIPTION].str[:26] + " " + products[Col.PRODUCT_SIZE],
            new_price=products[kpi_col(Scenario.RECOMMENDED, Metric.PRICE)],
        )
        return (
            alt.Chart(data)
            .mark_bar(cornerRadiusEnd=3, height={"band": 0.75})
            .encode(
                x=alt.X(
                    f"{Col.PRICE_RATIO}:Q",
                    title="Price change vs current",
                    axis=alt.Axis(format="%"),
                ),
                y=alt.Y("label:N", sort="-x", title=None),
                color=alt.Color(f"{Col.CATEGORY}:N", scale=self._category_scale, title="Category"),
                tooltip=[
                    alt.Tooltip(f"{Col.DESCRIPTION}:N", title="Product"),
                    alt.Tooltip(f"{Col.PRODUCT_SIZE}:N", title="Size"),
                    alt.Tooltip(f"{Col.ELASTICITY}:Q", title="Elasticity", format=".2f"),
                    alt.Tooltip(f"{Col.CURRENT_PRICE}:Q", title="Current price", format="$.2f"),
                    alt.Tooltip("new_price:Q", title="New price", format="$.2f"),
                    alt.Tooltip(f"{Col.PRICE_RATIO}:Q", title="Change", format="+.1%"),
                ],
            )
            .properties(height=max(14 * len(data), 200))
        )

    def objective_curve(
        self, curve: pd.DataFrame, marks: pd.DataFrame, goal: str
    ) -> alt.LayerChart:
        """Weekly goal value vs price, with current, own-optimum and recommended points.

        A muted red dotted line drops from the recommended point to the x-axis.
        `marks` holds columns label, price and value.
        """
        line = (
            alt.Chart(curve)
            .mark_line(color=self._style.series[0], strokeWidth=2)
            .encode(
                x=alt.X(f"{Col.PRICE}:Q", title="Regular price ($)", scale=alt.Scale(zero=False)),
                y=alt.Y(
                    f"{Col.OBJECTIVE_VALUE}:Q",
                    title=f"Weekly {goal} ($)",
                    scale=alt.Scale(zero=False),
                ),
                tooltip=[
                    alt.Tooltip(f"{Col.PRICE}:Q", title="Price", format="$.2f"),
                    alt.Tooltip(f"{Col.OBJECTIVE_VALUE}:Q", title=goal.title(), format="$,.0f"),
                ],
            )
        )
        points = (
            alt.Chart(marks)
            .mark_point(size=110, filled=True, stroke="white", strokeWidth=2)
            .encode(
                x="price:Q",
                y="value:Q",
                shape=alt.Shape("label:N", title=None),
                color=alt.Color(
                    "label:N",
                    title=None,
                    scale=alt.Scale(range=[self._style.ink, *self._style.series[1:3]]),
                ),
                tooltip=[
                    alt.Tooltip("label:N", title="Point"),
                    alt.Tooltip("price:Q", title="Price", format="$.2f"),
                    alt.Tooltip("value:Q", title=goal.title(), format="$,.0f"),
                ],
            )
        )
        drop = (
            alt.Chart(marks[marks["label"] == "Recommended"])
            .mark_rule(color="#D9534F", opacity=0.6, strokeWidth=1.5, strokeDash=[2, 3])
            .encode(x="price:Q", y="value:Q", y2=alt.value("height"))  # down to the x-axis
        )
        return (line + drop + points).properties(height=320)

    def scenarios(self, summary: pd.DataFrame) -> alt.Chart:
        """Volume, revenue and profit change vs current prices with no promo, per scenario."""
        order = [s for s in SCENARIO_LABELS if s in set(summary[Col.SCENARIO])]
        long = summary.melt(
            id_vars=[Col.SCENARIO, Col.TRADE_SPEND, Col.N_PROMOS],
            value_vars=[f"{m}_change" for m in METRIC_LABELS],
            var_name="metric",
            value_name="change",
        )
        long = long.assign(
            metric=long["metric"].str.removesuffix("_change").map(METRIC_LABELS),
            scenario=long[Col.SCENARIO].map(SCENARIO_LABELS),
        )
        labels = [SCENARIO_LABELS[s] for s in order]
        return (
            alt.Chart(long)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                x=alt.X(
                    "change:Q",
                    title="Change vs current prices, no promo",
                    axis=alt.Axis(format="%"),
                ),
                y=alt.Y("scenario:N", sort=labels, title=None),
                yOffset=alt.YOffset("metric:N", sort=list(METRIC_LABELS.values())),
                color=alt.Color(
                    "metric:N",
                    sort=list(METRIC_LABELS.values()),
                    title=None,
                    scale=alt.Scale(range=list(self._style.series[:3])),
                ),
                tooltip=[
                    alt.Tooltip("scenario:N", title="Scenario"),
                    alt.Tooltip("metric:N", title="Metric"),
                    alt.Tooltip("change:Q", title="Change", format="+.1%"),
                    alt.Tooltip(f"{Col.TRADE_SPEND}:Q", title="Trade spend", format="$,.0f"),
                    alt.Tooltip(f"{Col.N_PROMOS}:Q", title="Promos"),
                ],
            )
            .properties(height=60 * len(labels) + 40)
        )

    def calendar(self, weekly: pd.DataFrame) -> alt.LayerChart:
        """Promo calendar: promoted products x weeks, coloured by tactic, depth labelled."""
        promoted = weekly.groupby(Col.UPC)[Col.TACTIC].transform("count") > 0
        data = weekly[promoted].assign(
            label=weekly[Col.CATEGORY].str.title()
            + " | "
            + weekly[Col.DESCRIPTION].str[:22]
            + " "
            + weekly[Col.PRODUCT_SIZE],
            week_label=weekly[Col.WEEK].dt.strftime("%b %d"),
            tactic_label=weekly[Col.TACTIC].map(TACTIC_LABELS),
            depth_label=weekly[Col.DISCOUNT_DEPTH].map(lambda d: f"{d:.0%}" if pd.notna(d) else ""),
        )
        weeks = list(dict.fromkeys(data.sort_values(Col.WEEK)["week_label"]))
        rows = sorted(data["label"].unique())
        base = alt.Chart(data).encode(
            x=alt.X("week_label:O", sort=weeks, title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("label:N", sort=rows, title=None),
        )
        tactics = list(TACTIC_LABELS.values())
        cells = base.mark_rect(cornerRadius=3, stroke="white", strokeWidth=2).encode(
            color=alt.Color(
                "tactic_label:N",
                title="Tactic",
                scale=alt.Scale(domain=tactics, range=list(self._style.series[:4])),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip(f"{Col.DESCRIPTION}:N", title="Product"),
                alt.Tooltip("week_label:N", title="Week"),
                alt.Tooltip("tactic_label:N", title="Tactic"),
                alt.Tooltip("depth_label:N", title="Discount"),
                alt.Tooltip(f"{Col.PLAN_UNITS}:Q", title="Units (net of dip)", format=",.0f"),
                alt.Tooltip(f"{Col.FORECAST_BASELINE}:Q", title="Baseline units", format=",.0f"),
            ],
        )
        text = base.mark_text(fontSize=10, color=self._style.ink).encode(text="depth_label:N")
        return (cells + text).properties(height=max(26 * len(rows), 120))

    def frontier(self, frontier: pd.DataFrame, available: float, goal: str) -> alt.LayerChart:
        """Best incremental goal value vs trade budget; diamond = recommended budget."""
        line = (
            alt.Chart(frontier)
            .mark_line(point=alt.OverlayMarkDef(size=60), color=self._style.series[0])
            .encode(
                x=alt.X(f"{Col.BUDGET}:Q", title="Trade budget ($)", axis=alt.Axis(format="$,.0s")),
                y=alt.Y(
                    f"{Col.GAIN}:Q", title=f"Incremental {goal} ($)", axis=alt.Axis(format="$,.0s")
                ),
                tooltip=[
                    alt.Tooltip(f"{Col.BUDGET}:Q", title="Budget", format="$,.0f"),
                    alt.Tooltip(f"{Col.GAIN}:Q", title=f"Incremental {goal}", format="$,.0f"),
                    alt.Tooltip(f"{Col.N_PROMOS}:Q", title="Promos"),
                    alt.Tooltip(
                        f"{Col.MARGINAL_RETURN}:Q", title="Marginal return per $1", format=".2f"
                    ),
                ],
            )
        )
        best = (
            alt.Chart(frontier[frontier[Col.RECOMMENDED_BUDGET]])
            .mark_point(shape="diamond", size=220, filled=True, color=self._style.ink)
            .encode(x=f"{Col.BUDGET}:Q", y=f"{Col.GAIN}:Q")
        )
        cap = (
            alt.Chart(pd.DataFrame({"available": [available]}))
            .mark_rule(strokeDash=[4, 4], color=self._style.ink_secondary)
            .encode(
                x="available:Q",
                tooltip=[alt.Tooltip("available:Q", title="Your budget", format="$,.0f")],
            )
        )
        return (line + best + cap).properties(height=300)

    def weekly(self, weekly: pd.DataFrame) -> alt.Chart:
        """Weekly portfolio units: baseline at current and recommended prices, and with the plan."""
        totals = weekly.groupby(Col.WEEK)[list(FORECAST_LABELS)].sum().reset_index()
        long = totals.melt(id_vars=Col.WEEK, var_name="series", value_name="units")
        long["series"] = long["series"].map(FORECAST_LABELS)
        order = list(FORECAST_LABELS.values())
        return (
            alt.Chart(long)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X(f"{Col.WEEK}:T", title=None),
                y=alt.Y(
                    "units:Q", title="Units per week (all products)", scale=alt.Scale(zero=True)
                ),
                color=alt.Color(
                    "series:N",
                    sort=order,
                    title=None,
                    scale=alt.Scale(domain=order, range=list(self._style.series[:3])),
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip(f"{Col.WEEK}:T", title="Week"),
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("units:Q", title="Units", format=",.0f"),
                ],
            )
            .properties(height=300)
        )

    def backtest(self, backtest: pd.DataFrame) -> alt.Chart:
        """Holdout WAPE per forecast method; the method in use is fully opaque."""
        data = backtest.assign(
            method_label=backtest[Col.METHOD].str.replace("_", " ").str.capitalize(),
            used=backtest[Col.SELECTED].map({True: "Used for the horizon", False: "Challenger"}),
        )
        return (
            alt.Chart(data)
            .mark_bar(cornerRadiusEnd=3, color=self._style.series[0])
            .encode(
                x=alt.X(
                    f"{Col.WAPE}:Q",
                    title="Holdout WAPE (lower is better)",
                    axis=alt.Axis(format="%"),
                ),
                y=alt.Y("method_label:N", sort="x", title=None),
                opacity=alt.Opacity(
                    "used:N",
                    scale=alt.Scale(
                        domain=["Used for the horizon", "Challenger"], range=[1.0, 0.45]
                    ),
                    title=None,
                ),
                tooltip=[
                    alt.Tooltip("method_label:N", title="Method"),
                    alt.Tooltip(f"{Col.WAPE}:Q", title="WAPE", format=".1%"),
                    alt.Tooltip(f"{Col.BIAS}:Q", title="Bias", format="+.1%"),
                    alt.Tooltip(f"{Col.FIT_SECONDS}:Q", title="Fit seconds", format=".2f"),
                ],
            )
            .properties(height=180)
        )
