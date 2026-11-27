"""Price simulator page: re-optimize regular prices live, explore one product, edit prices.

Guardrails and margins are batched in a form; the NLopt solve (about 0.3 s) is
cached per setting. Edited prices are scored on the same demand curve.
"""

import numpy as np
import pandas as pd
import shared
import streamlit as st

from src.config import Config, Objective
from src.schema import Col, Metric, Scenario, kpi_col
from src.simulator import PriceSettings

directory = shared.directory()
goal = shared.goal()
simulator = shared.price_simulator(directory)
charts = shared.charts(directory)
defaults = PriceSettings.from_config(Config.load(), simulator.categories)
NEW_PRICE = kpi_col(Scenario.RECOMMENDED, Metric.PRICE)

shared.explain(
    f"Regular (non-promo) shelf prices that maximise **{goal}** for the base business, solved with "
    "NLopt SLSQP, rounded to prices ending in 9 and never worse than today's prices."
)

with st.form("price_settings"):
    st.markdown("**Guardrails**")
    c1, c2, c3 = st.columns(3)
    bounds = c1.slider(
        "Allowed price range (× current)",
        0.5,
        1.5,
        value=defaults.bounds,
        step=0.05,
        help="Each new price must stay within this multiple of its current price.",
    )
    floor = c2.slider(
        "Volume floor (share of current units)",
        0.80,
        1.00,
        value=defaults.volume_floor,
        step=0.01,
        help="Total weekly units across all products must stay at or above this share.",
    )
    cap = c3.slider(
        "Max average price increase",
        0.0,
        0.20,
        value=defaults.max_avg_increase,
        step=0.01,
        format="percent",
        help="Revenue-weighted average price change across the portfolio.",
    )
    ladder = st.toggle(
        "Keep branded products at least 10% above private label (per unit size)", value=True
    )
    st.markdown("**Category margins** (unit cost = (1 − margin) × current price)")
    margin_cols = st.columns(len(defaults.margins))
    margins = tuple(
        (
            category,
            col.number_input(
                category.title(), 0.05, 0.90, value=value, step=0.01, key=f"margin_{category}"
            ),
        )
        for col, (category, value) in zip(margin_cols, defaults.margins, strict=True)
    )
    st.form_submit_button("Re-optimize prices", icon=":material/calculate:", type="primary")

settings = PriceSettings((float(bounds[0]), float(bounds[1])), floor, cap, ladder, margins)
result = shared.optimize_prices(directory, goal, settings)
totals = result.totals

with st.grid(4, gap="medium"):
    for metric in (Metric.VOLUME, Metric.REVENUE, Metric.PROFIT):
        st.metric(
            f"Weekly {metric.value}",
            shared.money(totals.loc[metric.value, Scenario.RECOMMENDED.value])
            if metric != Metric.VOLUME
            else f"{totals.loc[metric.value, Scenario.RECOMMENDED.value]:,.0f} units",
            delta=f"{totals.loc[metric.value, 'change']:+.1%}",
            delta_description="vs current prices",
            border=True,
            height="stretch",
        )
    profit = totals.loc[Metric.PROFIT.value]
    weeks = shared.settings().annual_weeks
    annual = (profit[Scenario.RECOMMENDED.value] - profit[Scenario.CURRENT.value]) * weeks
    st.metric(
        "Annual profit impact",
        shared.money(annual),
        delta=f"{(result.products[Col.PRICE_RATIO].abs() > 0.005).sum()} prices change",
        delta_arrow="off",
        delta_color="gray",
        delta_description=f"of {len(result.products)} products",
        border=True,
        height="stretch",
    )
if result.safeguard_applied:
    st.warning(
        "Rounded prices scored below current prices, so current prices are kept.",
        title="Safeguard applied",
        icon=":material/shield:",
    )

with st.container(border=True):
    st.subheader("Recommended price changes")
    st.altair_chart(
        charts.price_changes(result.products),
        alt=f"Recommended regular-price change per product for the {goal} goal",
    )

st.subheader("Explore one product")
products = result.products.sort_values([Col.CATEGORY, Col.DESCRIPTION])
labels = dict(
    zip(
        products[Col.UPC], products[Col.DESCRIPTION] + " " + products[Col.PRODUCT_SIZE], strict=True
    )
)
picked = st.selectbox(
    "Product",
    options=list(labels),
    format_func=lambda u: labels[u],
    key="explore_upc",
    help="Shows the weekly goal value at every price for this product, holding others fixed.",
)
upc = picked if picked is not None else next(iter(labels))
row = products.set_index(Col.UPC).loc[upc]
curve = simulator.curve(Objective(goal), settings, int(upc))
own = curve[curve[Col.PRICE].between(*(np.array(bounds) * row[Col.CURRENT_PRICE]))]
own_price = own.loc[own[Col.OBJECTIVE_VALUE].idxmax(), Col.PRICE]
points = pd.DataFrame(
    {
        "label": ["Current", "Own optimum", "Recommended"],
        "price": [row[Col.CURRENT_PRICE], own_price, row[NEW_PRICE]],
    }
)
points["value"] = np.interp(points["price"], curve[Col.PRICE], curve[Col.OBJECTIVE_VALUE])
c1, c2 = st.columns([3, 1], gap="large")
with c1:
    st.altair_chart(
        charts.objective_curve(curve, points, goal),
        alt=f"Weekly {goal} versus price for {labels[upc]}",
    )
with c2:
    st.metric("Elasticity", f"{row[Col.ELASTICITY]:.2f}")
    st.metric("Current price", f"${row[Col.CURRENT_PRICE]:.2f}")
    st.metric("Recommended price", f"${row[NEW_PRICE]:.2f}", delta=f"{row[Col.PRICE_RATIO]:+.1%}")
    shared.explain(
        "The own optimum ignores portfolio guardrails; the recommendation can differ because the "
        "average-increase cap, volume floor and ladders trade moves across products."
    )

st.subheader("Edit prices yourself")
shared.explain("Change any new price and see the portfolio impact and which guardrails it breaks.")
editable = result.products[
    [
        Col.UPC,
        Col.DESCRIPTION,
        Col.PRODUCT_SIZE,
        Col.CATEGORY,
        Col.ELASTICITY,
        Col.CURRENT_PRICE,
        NEW_PRICE,
    ]
]
edited = st.data_editor(
    editable,
    key=f"price_editor_{goal}",
    hide_index=True,
    disabled=[c for c in editable.columns if c != NEW_PRICE],
    column_config={
        Col.UPC: None,
        Col.DESCRIPTION: st.column_config.TextColumn("Product"),
        Col.PRODUCT_SIZE: st.column_config.TextColumn("Size"),
        Col.CATEGORY: st.column_config.TextColumn("Category"),
        Col.ELASTICITY: st.column_config.NumberColumn("Elasticity", format="%.2f"),
        Col.CURRENT_PRICE: st.column_config.NumberColumn("Current price", format="$%.2f"),
        NEW_PRICE: st.column_config.NumberColumn("New price", format="$%.2f", min_value=0.01),
    },
    alt="Current and new regular price per product; the new price is editable",
)
scored = simulator.score(Objective(goal), settings, edited[NEW_PRICE].to_numpy())
with st.container(horizontal=True):
    for metric in (Metric.VOLUME, Metric.REVENUE, Metric.PROFIT):
        st.metric(
            f"Your prices: weekly {metric.value}",
            f"{scored.totals.loc[metric.value, 'change']:+.1%}",
            delta=round(
                scored.totals.loc[metric.value, "change"] - totals.loc[metric.value, "change"], 4
            ),
            format="percent",
            delta_description="vs the optimizer",
            border=True,
        )
if scored.broken:
    st.warning(
        ", ".join(b.replace("_", " ") for b in scored.broken),
        title="Guardrails broken",
        icon=":material/warning:",
    )
else:
    st.success("Your prices respect every guardrail.", icon=":material/check_circle:")
