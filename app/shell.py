"""App shell around every signed-in page: sidebar, pinned user bar and dialogs.

The sidebar is a flex column: a scrolling info block (goal, dataset facts, page
contents) and a user bar pinned to the bottom, whose popover opens Settings, the
Glossary page (methodology, architecture) and Sign out. Pinning needs a little CSS.
"""

from __future__ import annotations

from pathlib import Path

import shared
import streamlit as st

from src.config import Objective
from src.schema import Col
from src.simulator import SimulatorData
from src.simulator.accounts import User, UserSettings

LOGO = str(Path(__file__).parent / "assets" / "logo.svg")
INFO_KEY, USER_BAR_KEY = "sidebar_info", "user_bar"
GLOSSARY_PAGE = "app_pages/glossary.py"
MENU_KEYS: tuple[str, ...] = ("open_settings", "open_glossary", "sign_out")
BAR_COLORS: dict[str, dict[str, str]] = {
    "light": {"bar": "#F3F6FB", "rule": "#D9E1EE", "accent": "#1D5CD6",
              "hover": "rgba(29, 92, 214, 0.08)"},
    "dark": {"bar": "#111C2E", "rule": "#334155", "accent": "#2563EB",
             "hover": "rgba(96, 165, 250, 0.14)"},
}  # fmt: skip


def sidebar_css(theme_type: str | None) -> str:
    """CSS that pins the user bar: sidebar as a flex column, info block scrolls.

    Streamlit has no bottom-of-sidebar slot; rules target keyed containers
    (.st-key-*) and the sidebar's own test ids.
    """
    c = BAR_COLORS["dark" if theme_type == "dark" else "light"]
    menu = ", ".join(f".st-key-{k} button" for k in MENU_KEYS)
    return f"""
<style>
[data-testid="stSidebarContent"] {{ display: flex; flex-direction: column; overflow: hidden; }}
[data-testid="stSidebarHeader"] {{ flex: 0 0 auto; }}
[data-testid="stSidebarUserContent"] {{
    flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column;
    padding: 0; margin: 0 -1.25rem;
}}
[data-testid="stSidebarUserContent"] > div,
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {{
    flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column;
}}
[data-testid="stLayoutWrapper"]:has(> .st-key-{INFO_KEY}) {{
    flex: 1 1 auto; min-height: 0; overflow-y: auto;
}}
.st-key-{INFO_KEY} {{ padding: 0.75rem 1.25rem; }}
.st-key-{USER_BAR_KEY} {{
    padding: 0.4rem 0.75rem; background: {c["bar"]}; border-top: 1px solid {c["rule"]};
}}
.st-key-{USER_BAR_KEY} button, {menu} {{
    border: none; background: transparent; box-shadow: none;
    justify-content: flex-start; text-align: left;
}}
.st-key-{USER_BAR_KEY} button:hover, {", ".join(f".st-key-{k} button:hover" for k in MENU_KEYS)} {{
    background: {c["hover"]};
}}
.st-key-{USER_BAR_KEY} button > div, .st-key-{USER_BAR_KEY} button > div > span,
{", ".join(f".st-key-{k} button > div, .st-key-{k} button > div > span" for k in MENU_KEYS)} {{
    justify-content: flex-start;
}}
.st-key-{USER_BAR_KEY} button [data-testid="stMarkdownContainer"] {{
    text-align: left; width: 100%;
}}
.st-key-{USER_BAR_KEY} button {{ min-height: 0; padding: 0.25rem 0.5rem 0.25rem 0.25rem; }}
.st-key-{USER_BAR_KEY} button p {{ line-height: 1.25; margin: 0; }}
.st-key-{USER_BAR_KEY} button strong {{ font-size: 1rem; font-weight: 700; }}
.st-key-{USER_BAR_KEY} button > div > div:first-child > span {{
    align-items: stretch; gap: 0.75rem;
}}
.st-key-{USER_BAR_KEY} button > div > div:first-child > span > span:first-child {{
    align-self: stretch; align-items: center; width: auto; height: auto;
    padding: 0 0.55rem; border-radius: 0.5rem; background: {c["accent"]};
}}
.st-key-{USER_BAR_KEY} button > div > div:first-child [data-testid="stIconMaterial"] {{
    color: #FFFFFF;
}}
[data-testid="stPopoverBody"]:has(.st-key-open_settings) {{ padding: 0.25rem; }}
[data-testid="stPopoverBody"]:has(.st-key-open_settings) [data-testid="stVerticalBlock"] {{
    gap: 0;
}}
[data-testid="stPopoverBody"]:has(.st-key-open_settings) button {{
    min-height: 2.2rem; padding: 0.25rem 0.6rem;
}}
</style>
"""


PAGE_CONTENTS: dict[str, list[str]] = {
    "Overview": [
        "Headline results of the whole pipeline for the chosen goal.",
        "Regular-price impact, promo plan impact, recommended trade budget and forecast error.",
        "The 13-week scenarios: no promo, last year's calendar and the optimized plans.",
    ],
    "Price simulator": [
        "Re-optimizes regular prices live (NLopt) with your guardrails and category margins.",
        "Price change per product, and one product's profit or revenue curve.",
        "An editable price table that shows the impact and any broken guardrails.",
    ],
    "Promo planner": [
        "Re-plans the 13-week promo calendar live (MILP) for your budget and rules.",
        "Choose tactics, discount depths, fees, promos per product and display slots.",
        "Calendar, scenarios vs last year, weekly units and the budget frontier.",
    ],
    "Glossary": [
        "Methodology: what each step computes, why, its results and limitations.",
        "Architecture: code layout, data flow, extension points and the simulator design.",
    ],
    "Forecast": [
        "Backtest of four baseline forecasts: naive, seasonal index, SARIMAX and XGBoost.",
        "The 13-week weekly forecast with the pipeline's promo plan, by category.",
    ],
}


def sidebar(page_title: str, data: SimulatorData, user: User) -> None:
    """Scrolling info block (goal, dataset, page contents) and the pinned user bar."""
    st.logo(LOGO, size="large")
    st.html(sidebar_css(st.context.theme.type))
    with st.sidebar, st.container(key=INFO_KEY):
        st.segmented_control(
            "Goal",
            options=data.objectives,
            required=True,
            format_func=str.title,
            key=shared.GOAL_KEY,
            persist_state="session",
            help="Profit maximises margin; revenue maximises sales value. Every page uses it.",
        )
        _dataset(data)
        st.markdown("**What this page contains**")
        st.caption("\n".join(f"- {line}" for line in PAGE_CONTENTS.get(page_title, [])))
    _user_bar(user)


def _dataset(data: SimulatorData) -> None:
    """Static facts about the dataset and the planning horizon (muted, small)."""
    products = data.recommendations.drop_duplicates(Col.UPC)
    weeks = data.forecast[Col.WEEK]
    categories = ", ".join(sorted(products[Col.CATEGORY].str.title().unique()))
    st.markdown("**About the data**")
    st.caption(
        "- dunnhumby *Breakfast at the Frat*: weekly store sales, Jan 2009 to Jan 2012\n"
        f"- {len(products)} products in {products[Col.CATEGORY].nunique()} categories: "
        f"{categories}\n"
        f"- Forecast and promo horizon: {weeks.min():%d %b %Y} to {weeks.max():%d %b %Y}\n"
        "- Costs, margins and promo fees are assumptions (no cost data in the dataset)"
    )


def _user_bar(user: User) -> None:
    """Pinned bar: name and email open a popover with Settings, Glossary and Sign out."""
    with st.sidebar, st.container(key=USER_BAR_KEY):
        label = f"**{user.name}**  \n:small[:gray[{user.email}]]"
        with st.popover(label, icon=":material/account_circle:", width="stretch"):
            settings = st.button(
                "Settings", icon=":material/settings:", key="open_settings", width="stretch"
            )
            glossary = st.button(
                "Glossary", icon=":material/menu_book:", key="open_glossary", width="stretch"
            )
            leave = st.button("Sign out", icon=":material/logout:", key="sign_out", width="stretch")
    if settings:
        settings_dialog(user)
    elif glossary:
        st.switch_page(GLOSSARY_PAGE)
    elif leave:
        shared.sign_out()
        st.rerun()


@st.dialog("Settings", icon=":material/settings:", width="medium")
def settings_dialog(user: User) -> None:
    """Edit and save the signed-in user's preferences."""
    current = shared.settings()
    with st.form("settings_form", border=False):
        goal = st.segmented_control(
            "Goal at sign-in",
            key="settings_goal",
            options=[o.value for o in (Objective.PROFIT, Objective.REVENUE)],
            default=current.default_goal,
            required=True,
            format_func=str.title,
            help="The goal selected when you sign in. Switch it any time in the sidebar.",
        )
        weeks = st.number_input(
            "Weeks per year",
            key="settings_weeks",
            min_value=48,
            max_value=53,
            value=current.annual_weeks,
            help="Turns weekly profit changes into the annual impact on the Price simulator.",
        )
        points = st.slider(
            "Budget frontier points",
            key="settings_points",
            min_value=4,
            max_value=30,
            value=current.frontier_points,
            help="How many budgets the Promo planner solves when tracing the frontier.",
        )
        hurdle = st.number_input(
            "Minimum marginal return per $1",
            key="settings_hurdle",
            min_value=0.0,
            max_value=3.0,
            value=current.min_marginal_return,
            step=0.05,
            format="%.2f",
            help="The recommended budget is the last step returning at least this much per "
            "extra $1 of trade spend.",
        )
        explain = st.toggle(
            "Show explanations under charts",
            key="settings_explain",
            value=current.show_explanations,
        )
        st.caption(
            "Light or dark mode: open the :material/more_vert: menu at the top right, then "
            "Settings."
        )
        applied = st.form_submit_button("Apply", icon=":material/check:", type="primary")
    if applied:
        updated = UserSettings(str(goal), int(weeks), int(points), float(hurdle), bool(explain))
        shared.store().save_settings(user.email, updated)
        st.session_state[shared.SETTINGS_KEY] = updated
        st.rerun()


@st.dialog("About the tool", icon=":material/info:", width="medium")
def about_dialog() -> None:
    """Short description of the simulator for the sign-in page."""
    st.markdown(
        "A what-if simulator on top of an end-to-end Revenue Growth Management pipeline:\n\n"
        "- **Price elasticity** from four models, up to a Bayesian hierarchical model.\n"
        "- **Baseline** volume without promotions, and a backtested 13-week forecast.\n"
        "- **Regular prices** re-optimized live with NLopt within your guardrails.\n"
        "- **Promotions** re-planned live as a MILP: product, week, tactic, depth and budget.\n\n"
        "Accounts are local to this machine. Create one in a terminal with `make user`."
    )
