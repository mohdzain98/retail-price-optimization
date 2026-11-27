"""Overview page: headline results of the pipeline for the chosen goal.

Reads the stored Phase 4 and Phase 5 outputs (no re-solving) and links to the
simulator pages, where the optimizers re-run live.
"""

import shared
import streamlit as st

from src.config import Config
from src.schema import Col, PromoScenario, Scenario

directory = shared.directory()
data = shared.data(directory)
goal = shared.goal()
charts = shared.charts(directory)

shared.explain(
    "An end-to-end Revenue Growth Management pipeline: Bayesian price elasticity, a top-down "
    "baseline, NLopt regular-price optimization, a backtested baseline forecast and a MILP promo "
    f"calendar under a trade budget. Showing results for the **{goal}** goal."
)

opt = data.optimization_summary
price = opt[(opt[Col.OBJECTIVE] == goal) & (opt[Col.SCENARIO] == Scenario.RECOMMENDED.value)].iloc[
    0
]
promo = data.promo_summary[data.promo_summary[Col.OBJECTIVE] == goal].set_index(Col.SCENARIO)
milp, last_year = (
    promo.loc[PromoScenario.MILP_PLAN.value],
    promo.loc[PromoScenario.LAST_YEAR_PLAN.value],
)
frontier = data.frontier[data.frontier[Col.OBJECTIVE] == goal]
budget = frontier.loc[frontier[Col.RECOMMENDED_BUDGET], Col.BUDGET].iloc[0]
best = data.backtest[data.backtest[Col.SELECTED]].iloc[0]
weekly_profit = price["annual_profit_delta"] / Config.load().optimization.annualize_weeks
annual_profit = weekly_profit * shared.settings().annual_weeks

with st.grid(4, gap="medium"):
    st.metric(
        f"Regular prices: weekly {goal}",
        f"{price[f'{goal}_change']:+.1%}",
        delta=f"{shared.money(annual_profit)} profit per year",
        delta_description="recommended vs current prices",
        border=True,
        height="stretch",
    )
    st.metric(
        f"Promo plan: 13-week {goal}",
        f"{milp[f'{goal}_change']:+.1%}",
        delta=f"{shared.money(milp[Col.PROFIT] - last_year[Col.PROFIT])} profit",
        delta_description="vs last year's calendar",
        border=True,
        height="stretch",
    )
    st.metric(
        "Recommended trade budget",
        shared.money(budget),
        delta=f"{shared.money(last_year[Col.TRADE_SPEND] - milp[Col.TRADE_SPEND])} less",
        delta_description="than last year's spend",
        border=True,
        height="stretch",
    )
    st.metric(
        "Baseline forecast error",
        f"{best[Col.WAPE]:.1%} WAPE",
        delta=f"{best[Col.METHOD].replace('_', ' ')} wins",
        delta_arrow="off",
        delta_color="gray",
        delta_description="of 4 methods on the holdout",
        border=True,
        height="stretch",
    )

left, right = st.columns([2, 3], gap="large")
with left:
    st.subheader("How it works")
    st.markdown(
        "1. **Elasticity:** four models (OLS to Bayesian hierarchical) estimate how volume "
        "responds to regular and promo prices.\n"
        "2. **Baseline:** weekly volume without promotions, from the elasticity model.\n"
        "3. **Regular prices:** NLopt maximises the goal within price, volume, average-increase "
        "and price-ladder guardrails.\n"
        "4. **Forecast:** the baseline for the next 13 weeks, from the best of four methods.\n"
        "5. **Promotions:** a MILP picks product, week, tactic and depth under a trade budget; "
        "a budget frontier picks the budget itself."
    )
    st.page_link(
        "app_pages/prices.py", label="Try your own pricing guardrails", icon=":material/sell:"
    )
    st.page_link(
        "app_pages/promos.py", label="Plan promotions with your budget", icon=":material/campaign:"
    )
with right, st.container(border=True):
    st.subheader("13-week scenarios")
    st.altair_chart(
        charts.scenarios(promo.reset_index()),
        alt=f"Volume, revenue and profit change by scenario for the {goal} goal",
    )
    shared.explain("All plans use the recommended regular prices. Profit is net of trade spend.")
