"""Cached loaders and helpers shared by the app pages.

Pipeline outputs are loaded once per data directory; simulations are cached on
their (hashable) settings so moving back to an earlier setting is instant.
Imported by pages as `import shared` (the entry point's folder is on sys.path).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import Config, Objective
from src.plots.style import PlotStyle
from src.simulator import (
    PriceResult,
    PriceSettings,
    PriceSimulator,
    PromoResult,
    PromoSettings,
    PromoSimulator,
    SimulatorCharts,
    SimulatorData,
)
from src.simulator.accounts import User, UserSettings, UserStore

GOAL_KEY = "goal"
USER_KEY = "user"
SETTINGS_KEY = "settings"
SIGNED_OUT_KEY = "signed_out"


@st.cache_resource(show_spinner="Loading pipeline outputs...")
def data(directory: str) -> SimulatorData:
    """All processed tables for one data directory (shared by every session)."""
    return SimulatorData.load(Path(directory))


def directory() -> str:
    """Data directory as a string cache key."""
    return str(SimulatorData.directory())


@st.cache_resource
def price_simulator(directory: str) -> PriceSimulator:
    """Price simulator over the stored recommendations."""
    return PriceSimulator(Config.load(), data(directory).recommendations)


@st.cache_resource
def promo_simulator(directory: str) -> PromoSimulator:
    """Promo simulator over the stored options."""
    return PromoSimulator(data(directory))


@st.cache_data(max_entries=64, show_spinner="Re-optimizing prices...")
def optimize_prices(directory: str, objective: str, settings: PriceSettings) -> PriceResult:
    """NLopt solve for one goal and set of guardrails."""
    return price_simulator(directory).optimize(Objective(objective), settings)


@st.cache_data(max_entries=32, show_spinner="Planning promotions (MILP)...")
def plan_promos(directory: str, objective: str, settings: PromoSettings) -> PromoResult:
    """MILP promo plan for one goal and set of settings."""
    return promo_simulator(directory).plan(objective, settings)


@st.cache_data(max_entries=16, show_spinner="Tracing the budget frontier...")
def promo_frontier(
    directory: str, objective: str, settings: PromoSettings, points: int, hurdle: float
) -> pd.DataFrame:
    """Budget frontier (several MILP solves) for one goal and set of settings."""
    return promo_simulator(directory).frontier(objective, settings, points, hurdle)


def charts(directory: str) -> SimulatorCharts:
    """Chart builder with the project palette."""
    return SimulatorCharts(PlotStyle(), price_simulator(directory).categories)


@st.cache_resource
def user_store(path: str) -> UserStore:
    """Account store for one SQLite file."""
    return UserStore(Path(path))


def store() -> UserStore:
    """Account store at the configured path (RPO_USERS_DB or data/users.db)."""
    return user_store(str(UserStore.default_path()))


def user() -> User | None:
    """The signed-in user of this session, if any."""
    current = st.session_state.get(USER_KEY)
    return current if isinstance(current, User) else None


def settings() -> UserSettings:
    """The signed-in user's settings (defaults before sign-in)."""
    current = st.session_state.get(SETTINGS_KEY)
    return current if isinstance(current, UserSettings) else UserSettings()


def sign_in(account: User) -> None:
    """Start a session for `account` and apply its saved settings."""
    prefs = store().settings(account.email)
    st.session_state[USER_KEY] = account
    st.session_state[SETTINGS_KEY] = prefs
    st.session_state[GOAL_KEY] = prefs.default_goal


def sign_out() -> None:
    """End the session: clear all session state and flag the sign-out for one run."""
    st.session_state.clear()
    st.session_state[SIGNED_OUT_KEY] = True


def consume_signed_out() -> bool:
    """True once right after a sign-out (the flag is removed when read)."""
    return bool(st.session_state.pop(SIGNED_OUT_KEY, False))


def explain(text: str) -> None:
    """Explanatory caption, shown unless the user turned explanations off."""
    if settings().show_explanations:
        st.caption(text)


def goal() -> str:
    """The goal chosen in the sidebar (profit or revenue)."""
    return str(st.session_state.get(GOAL_KEY) or Objective.PROFIT.value)


def money(value: float) -> str:
    """Compact dollar label, e.g. $72.4k or -$1.2M."""
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1e6:
        return f"{sign}${value / 1e6:,.1f}M"
    if value >= 1e3:
        return f"{sign}${value / 1e3:,.1f}k"
    return f"{sign}${value:,.0f}"
