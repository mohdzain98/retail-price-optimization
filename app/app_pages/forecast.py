"""Forecast page: baseline forecast backtest and the 13-week weekly forecast.

Shows the stored backtest of the four methods and the pipeline's forecast with
its MILP promo plan, filterable by category or product.
"""

import shared
import streamlit as st

from src.schema import Col

directory = shared.directory()
goal = shared.goal()
data = shared.data(directory)
charts = shared.charts(directory)
backtest = data.backtest
used = backtest[backtest[Col.SELECTED]].iloc[0]

shared.explain(
    "The planner needs the **baseline** (volume without promotions) for each future week. Four "
    "methods forecast it for the 13 holdout weeks from the weeks before them; the lowest error "
    "is used for the horizon."
)

left, right = st.columns([3, 2], gap="large")
with left, st.container(border=True):
    st.subheader("Backtest on the 13 holdout weeks")
    st.altair_chart(
        charts.backtest(backtest), alt="Holdout WAPE of the four baseline forecast methods"
    )
with right:
    st.metric(
        "Method used",
        used[Col.METHOD].replace("_", " ").capitalize(),
        delta=f"{used[Col.WAPE]:.1%} WAPE",
        delta_arrow="off",
        delta_color="gray",
        delta_description=f"bias {used[Col.BIAS]:+.1%}",
        border=True,
    )
    st.markdown(
        "- **Naive flat:** the last 13 weeks' average.\n"
        "- **Seasonal index:** that level × the Bayesian model's Fourier and holiday effects.\n"
        "- **SARIMAX(0,1,1):** per product, local level plus calendar regressors.\n"
        "- **XGBoost:** one model for all products, forecasting 1–13 weeks ahead."
    )

st.subheader(f"13-week forecast (goal: {goal})")
forecast = data.forecast[data.forecast[Col.OBJECTIVE] == goal]
categories = sorted(forecast[Col.CATEGORY].unique())
chosen = st.pills(
    "Categories",
    options=categories,
    default=categories,
    selection_mode="multi",
    format_func=str.title,
    key="forecast_categories",
)
view = forecast[forecast[Col.CATEGORY].isin(chosen)]
if view.empty:
    st.info("Pick at least one category.", icon=":material/info:")
    st.stop()
with st.container(border=True):
    st.altair_chart(
        charts.weekly(view),
        alt="Weekly units: baseline at current and new prices, and with the pipeline's promo plan",
    )
    shared.explain(
        "With promo plan: the pipeline's MILP plan at the recommended budget, including the "
        "post-promo dip in the following weeks."
    )
