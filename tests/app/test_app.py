"""Tests for the Streamlit app (app/streamlit_app.py and app_pages/) with AppTest.

Pages run headless against synthetic pipeline outputs (RPO_DATA_DIR) and a
temporary account database (RPO_USERS_DB): sign-in, every page for both goals,
the promo form, the account menu (settings, glossary, sign out) and missing data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.simulator.accounts import USERS_DB_ENV, User, UserSettings, UserStore
from src.simulator.data import DATA_DIR_ENV

APP = str(Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py")
PAGES = ("app_pages/overview.py", "app_pages/prices.py", "app_pages/promos.py",
         "app_pages/forecast.py")  # fmt: skip
TIMEOUT = 120
EMAIL, PASSWORD = "jane@example.com", "secret123"


@pytest.fixture
def env(processed_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> UserStore:
    """Synthetic outputs plus a fresh account database with one user."""
    monkeypatch.setenv(DATA_DIR_ENV, str(processed_dir))
    monkeypatch.setenv(USERS_DB_ENV, str(tmp_path / "users.db"))
    store = UserStore(tmp_path / "users.db")
    store.create("Jane Doe", EMAIL, PASSWORD)
    return store


def _signed_in(goal: str = "profit") -> AppTest:
    """App with a signed-in session (as the sign-in page would leave it)."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.session_state["user"] = User("Jane Doe", EMAIL)
    at.session_state["settings"] = UserSettings(default_goal=goal)
    at.session_state["goal"] = goal
    return at.run()


class TestSignIn:
    """Signed-out sessions only see the sign-in page."""

    def test_wrong_password(self, env: UserStore) -> None:
        """A wrong password shows an error and keeps the user out."""
        at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
        at.text_input(key="login_email").input(EMAIL)
        at.text_input(key="login_password").input("wrong1234")
        at.button(key="FormSubmitter:sign_in-Sign in").click().run()
        assert "incorrect" in at.error[0].value
        assert "user" not in at.session_state

    def test_sign_in(self, env: UserStore) -> None:
        """Correct details open the Overview with the account in the sidebar."""
        at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
        at.text_input(key="login_email").input(EMAIL.upper())
        at.text_input(key="login_password").input(PASSWORD)
        at.button(key="FormSubmitter:sign_in-Sign in").click().run()
        assert not at.exception, at.exception
        assert at.title[0].value == "Overview"
        assert at.session_state["user"].email == EMAIL

    def test_no_accounts_hint(
        self, processed_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no account yet, the page says to run `make user`."""
        monkeypatch.setenv(USERS_DB_ENV, str(tmp_path / "empty.db"))
        at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
        assert "make user" in at.info[0].value


class TestPages:
    """Every page renders for both goals without exceptions."""

    @pytest.mark.parametrize("page", PAGES)
    @pytest.mark.parametrize("goal", ["profit", "revenue"])
    def test_pages(self, env: UserStore, page: str, goal: str) -> None:
        """The page runs, shows metrics and the sidebar shell."""
        at = _signed_in(goal)
        at.switch_page(page).run()
        assert not at.exception, at.exception
        assert at.metric, "every page shows at least one metric"
        assert any("What this page contains" in m.value for m in at.sidebar.markdown)

    def test_promo_form(self, env: UserStore) -> None:
        """Submitting a zero budget plans no promotions."""
        at = _signed_in()
        at.switch_page("app_pages/promos.py").run()
        at.number_input(key="budget_profit").set_value(0)
        at.button(key="FormSubmitter:promo_settings-Plan promotions").click().run()
        assert not at.exception, at.exception
        assert next(m for m in at.metric if m.label == "Promotions").value == "0"

    def test_missing_outputs(
        self, env: UserStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty data directory shows a clear error after sign-in."""
        monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "no_outputs"))
        at = _signed_in()
        assert not at.exception
        assert "make optimize" in at.error[0].value


class TestAccountMenu:
    """Settings save per user; sign out returns to the sign-in page."""

    def test_settings_saved(self, env: UserStore) -> None:
        """Applying settings stores them and hides explanations."""
        at = _signed_in()
        at.button(key="open_settings").click().run()
        at.toggle(key="settings_explain").set_value(False)
        at.number_input(key="settings_weeks").set_value(50)
        # AppTest reruns the whole script (browsers rerun only the dialog), so re-open it
        at.button(key="open_settings").click()
        at.button(key="FormSubmitter:settings_form-Apply").click().run()
        assert not at.exception, at.exception
        saved = env.settings(EMAIL)
        assert saved.annual_weeks == 50 and not saved.show_explanations

    def test_glossary(self, env: UserStore) -> None:
        """The user bar opens the Glossary page with both docs as tabs."""
        at = _signed_in()
        at.button(key="open_glossary").click().run()
        assert not at.exception, at.exception
        assert at.title[0].value == "Glossary"
        assert len(at.tabs) == 2 and len(at.markdown) > 2

    def test_sign_out(self, env: UserStore) -> None:
        """Sign out clears the session and returns to the sign-in page."""
        at = _signed_in()
        at.button(key="sign_out").click().run()
        assert "user" not in at.session_state
        assert at.button(key="FormSubmitter:sign_in-Sign in")
