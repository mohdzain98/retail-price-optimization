"""Streamlit entry point of the retail price and promo simulator (Phase 6).

Signed-out sessions see only the sign-in page (accounts from `make user`).
Signed-in sessions get the pipeline outputs, the sidebar shell (goal, dataset,
page contents, account menu) and the four pages in app_pages/. Run with `make app`.
"""

import shared
import shell
import streamlit as st

from src.utils import MissingFileError

user = shared.user()
st.set_page_config(
    page_title="Retail price & promo simulator",
    page_icon=":material/storefront:",
    layout="wide",
    initial_sidebar_state="collapsed" if user is None else "expanded",  # no sidebar when signed out
)
if user is None:
    # Signed out: nothing is drawn in the sidebar, but Streamlit keeps an empty sidebar
    # shell and its collapse arrow after a sign-out, so hide them.
    st.html(
        '<style>[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] '
        "{ display: none; }</style>"
    )
    landing = st.Page("app_pages/login.py", title="Sign in", icon=":material/lock:")
    navigation = st.navigation([landing], position="hidden")
    if shared.consume_signed_out():
        st.switch_page(landing)  # so the URL no longer shows the last page
    navigation.run()
    st.stop()

try:
    data = shared.data(shared.directory())
except MissingFileError as err:
    st.error(str(err), title="Pipeline outputs not found", icon=":material/error:")
    st.stop()

page = st.navigation(
    [
        st.Page(
            "app_pages/overview.py",
            title="Overview",
            icon=":material/dashboard:",
            help="Headline results of the pipeline for the chosen goal, and how each step works.",
        ),
        st.Page(
            "app_pages/prices.py",
            title="Price simulator",
            icon=":material/sell:",
            help="Re-optimize regular prices with your own guardrails and margins, or edit prices "
            "and see the impact.",
        ),
        st.Page(
            "app_pages/promos.py",
            title="Promo planner",
            icon=":material/campaign:",
            help="Re-plan the 13-week promo calendar with the MILP: budget, tactics, depths, fees "
            "and calendar rules.",
        ),
        st.Page(
            "app_pages/forecast.py",
            title="Forecast",
            icon=":material/insights:",
            help="Baseline forecast backtest (naive, seasonal index, SARIMAX, XGBoost) and the "
            "13-week weekly forecast.",
        ),
        st.Page(
            shell.GLOSSARY_PAGE,
            title="Glossary",
            icon=":material/menu_book:",
            url_path="glossary",
            visibility="hidden",
            help="Methodology and architecture docs. Opened from the user bar.",
        ),
    ],
    position="top",
)
shell.sidebar(page.title, data, user)
st.title(page.title)
page.run()
