"""Promo planner page: re-plan the 13-week promo calendar with the MILP, live.

Budget, tactics, depths, fees and calendar rules are batched in a form; the plan
(well under a second) is cached per setting. The budget frontier runs on request.
"""

import dataclasses

import shared
import streamlit as st

from src.config import Config
from src.schema import Col, PromoScenario
from src.simulator import PromoSettings
from src.simulator.charts import TACTIC_LABELS
from src.simulator.promo import YOUR_PLAN

directory = shared.directory()
goal = shared.goal()
data = shared.data(directory)
charts = shared.charts(directory)
cfg = Config.load()
defaults = PromoSettings.from_config(cfg)
frontier = data.frontier[data.frontier[Col.OBJECTIVE] == goal]
recommended_budget = float(frontier.loc[frontier[Col.RECOMMENDED_BUDGET], Col.BUDGET].iloc[0])

shared.explain(
    f"Which product to promote, in which week, with which tactic and discount, to maximise "
    f"**{goal}** under a trade budget. Solved as a MILP with HiGHS; promo lifts come from the "
    "Bayesian model at the recommended regular prices."
)

with st.form("promo_settings"):
    c1, c2 = st.columns(2, gap="large")
    with c1:
        budget = st.number_input(
            "Trade budget over 13 weeks ($)",
            0,
            1_000_000,
            value=int(round(recommended_budget, -2)),
            step=5_000,
            key=f"budget_{goal}",
            help="Discount funding plus fixed fees. Defaults to the pipeline's recommended budget.",
        )
        tactics = st.pills(
            "Tactics",
            options=list(defaults.tactics),
            default=list(defaults.tactics),
            selection_mode="multi",
            format_func=lambda t: TACTIC_LABELS[t],
        )
        depths = st.pills(
            "Discount depths",
            options=list(defaults.depths),
            default=list(defaults.depths),
            selection_mode="multi",
            format_func=lambda d: f"{d:.0%}",
        )
    with c2:
        r1, r2 = st.columns(2)
        max_promos = r1.slider("Max promos per product", 1, 13, value=defaults.max_promos)
        gap = r2.slider(
            "Min weeks between promos",
            0,
            6,
            value=defaults.min_gap_weeks,
            help="At least this many weeks between two promos of one product (pantry loading).",
        )
        feature_slots = r1.number_input(
            "Feature slots per week", 0, 60, value=defaults.feature_slots
        )
        display_slots = r2.number_input(
            "Display slots per week", 0, 60, value=defaults.display_slots
        )
    st.markdown("**Fixed fee per product-week** (assumed; the data has no trade-spend figures)")
    fee_cols = st.columns(len(defaults.fees))
    fees = tuple(
        (
            tactic,
            float(
                col.number_input(
                    TACTIC_LABELS[tactic], 0, 5_000, value=int(fee), step=50, key=f"fee_{tactic}"
                )
            ),
        )
        for col, (tactic, fee) in zip(fee_cols, defaults.fees, strict=True)
    )
    st.form_submit_button("Plan promotions", icon=":material/event_available:", type="primary")

if not tactics or not depths:
    st.info("Pick at least one tactic and one discount depth.", icon=":material/info:")
    st.stop()

settings = dataclasses.replace(
    defaults,
    budget=float(budget),
    max_promos=max_promos,
    min_gap_weeks=gap,
    feature_slots=int(feature_slots),
    display_slots=int(display_slots),
    tactics=tuple(tactics),
    depths=tuple(depths),
    fees=fees,
)
result = shared.plan_promos(directory, goal, settings)
summary = result.summary.set_index(Col.SCENARIO)
plan, last_year = summary.loc[YOUR_PLAN], summary.loc[PromoScenario.LAST_YEAR_PLAN.value]
regular = summary.loc[PromoScenario.RECOMMENDED_NO_PROMO.value]

with st.grid(4, gap="medium"):
    st.metric(
        f"13-week {goal} with your plan",
        f"{plan[f'{goal}_change']:+.1%}",
        delta=f"{plan[f'{goal}_change'] - regular[f'{goal}_change']:+.1%} pts",
        delta_description="from promotions alone",
        border=True,
        height="stretch",
    )
    st.metric(
        "Profit vs last year's calendar",
        shared.money(plan[Col.PROFIT] - last_year[Col.PROFIT]),
        delta=f"{plan[Col.PROFIT] / last_year[Col.PROFIT] - 1:+.1%}",
        delta_description="same weeks, same model",
        border=True,
        height="stretch",
    )
    st.metric(
        "Trade spend",
        shared.money(plan[Col.TRADE_SPEND]),
        delta=f"{shared.money(budget - plan[Col.TRADE_SPEND])} unspent",
        delta_arrow="off",
        delta_color="gray",
        delta_description="more promos would not pay back"
        if budget - plan[Col.TRADE_SPEND] > 1_000
        else "budget fully used",
        border=True,
        height="stretch",
    )
    st.metric(
        "Promotions",
        f"{int(plan[Col.N_PROMOS])}",
        delta=f"{int(last_year[Col.N_PROMOS])} last year",
        delta_arrow="off",
        delta_color="gray",
        delta_description=f"on {result.plan[Col.UPC].nunique()} products",
        border=True,
        height="stretch",
    )

with st.container(border=True):
    st.subheader("Promo calendar")
    if result.plan.empty:
        st.info("No promotion pays back under these settings.", icon=":material/info:")
    else:
        st.altair_chart(
            charts.calendar(result.weekly),
            alt=f"Promo calendar of the MILP plan for the {goal} goal: product by week, by tactic",
        )
        shared.explain(
            "Cell = discount depth. Hover a cell for promo-week units, net of the post-promo dip."
        )

left, right = st.columns(2, gap="large")
with left, st.container(border=True):
    st.subheader("Scenarios")
    st.altair_chart(
        charts.scenarios(result.summary),
        alt="Volume, revenue and profit change for no promo, last year's calendar and your plan",
    )
with right, st.container(border=True):
    st.subheader("Weekly units")
    st.altair_chart(
        charts.weekly(result.weekly),
        alt="Weekly portfolio units: baseline at current and new prices, and with your plan",
    )

with st.container(border=True):
    st.subheader("Budget frontier")
    show = st.toggle(
        "Trace the frontier for these settings",
        key="show_frontier",
        help="Solves the MILP at several budgets, from zero up to the unconstrained plan's spend.",
    )
    if show:
        prefs = shared.settings()
        traced = shared.promo_frontier(
            directory, goal, settings, prefs.frontier_points, prefs.min_marginal_return
        )
        st.altair_chart(
            charts.frontier(traced, settings.budget, goal),
            alt=f"Best incremental {goal} at each trade budget, with diminishing returns",
        )
        best = traced[traced[Col.RECOMMENDED_BUDGET]].iloc[0]
        shared.explain(
            f"Diamond: the recommended budget, {shared.money(best[Col.BUDGET])}, the last step "
            f"returning at least ${prefs.min_marginal_return:.2f} of {goal} per extra $1 "
            "(capped at your budget). Dashed line: your budget."
        )

st.download_button(
    "Download plan (CSV)",
    data=result.plan.to_csv(index=False),
    file_name=f"promo_plan_{goal}.csv",
    mime="text/csv",
    icon=":material/download:",
    disabled=result.plan.empty,
)
