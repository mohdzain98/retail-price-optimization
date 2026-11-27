"""Tests for src.simulator.accounts: password hashing, validation, store and settings.

Each test uses its own SQLite file under tmp_path. One test class per production
class.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.simulator.accounts import (
    AccountValidator,
    PasswordHasher,
    User,
    UserSettings,
    UserStore,
)
from src.utils import AccountError

PASSWORD = "secret123"


@pytest.fixture
def store(tmp_path: Path) -> UserStore:
    """Empty store in a temporary file."""
    return UserStore(tmp_path / "users.db")


class TestPasswordHasher:
    """Hashes are salted and verify only the right password."""

    def test_round_trip(self) -> None:
        """The right password verifies, a wrong one does not."""
        hasher = PasswordHasher()
        stored = hasher.hash(PASSWORD)
        assert hasher.verify(PASSWORD, stored)
        assert not hasher.verify("secret124", stored)

    def test_salted(self) -> None:
        """Two hashes of one password differ and never contain it."""
        hasher = PasswordHasher()
        first, second = hasher.hash(PASSWORD), hasher.hash(PASSWORD)
        assert first != second and PASSWORD not in first

    @pytest.mark.parametrize("stored", ["", "plain", "md5$1$2$3$ab$cd", "scrypt$x$8$1$zz$zz"])
    def test_malformed(self, stored: str) -> None:
        """Malformed or foreign hashes never verify."""
        assert not PasswordHasher().verify(PASSWORD, stored)


class TestAccountValidator:
    """Inputs are normalised or rejected with a reason."""

    def test_normalises(self) -> None:
        """Email lower-cased and trimmed; name whitespace collapsed."""
        rules = AccountValidator()
        assert rules.email("  Jane.Doe@Gmail.com ") == "jane.doe@gmail.com"
        assert rules.name("  Jane   Doe ") == "Jane Doe"

    @pytest.mark.parametrize("email", ["", "jane", "jane@", "jane@gmail", "a b@gmail.com"])
    def test_bad_email(self, email: str) -> None:
        """Invalid emails are rejected."""
        with pytest.raises(AccountError, match="valid email"):
            AccountValidator().email(email)

    @pytest.mark.parametrize("password", ["short1", "lettersonly", "12345678"])
    def test_weak_password(self, password: str) -> None:
        """Too short, no digit or no letter is rejected."""
        with pytest.raises(AccountError, match="Password"):
            AccountValidator().password(password)

    def test_empty_name(self) -> None:
        """A blank name is rejected."""
        with pytest.raises(AccountError, match="Name"):
            AccountValidator().name("   ")


class TestUserStore:
    """Accounts are created once and authenticate by email and password."""

    def test_create_and_authenticate(self, store: UserStore) -> None:
        """Sign-in works with any email casing; wrong password or email fails."""
        created = store.create("Jane Doe", "Jane@Example.com", PASSWORD)
        assert created == User("Jane Doe", "jane@example.com")
        assert store.authenticate(" JANE@example.com", PASSWORD) == created
        assert store.authenticate("jane@example.com", "wrong1234") is None
        assert store.authenticate("nobody@example.com", PASSWORD) is None
        assert store.count() == 1 and store.exists("JANE@EXAMPLE.COM")

    def test_duplicate(self, store: UserStore) -> None:
        """A second account with the same email is refused."""
        store.create("Jane", "jane@example.com", PASSWORD)
        with pytest.raises(AccountError, match="already exists"):
            store.create("Other", "JANE@example.com", PASSWORD)
        assert store.count() == 1

    def test_password_not_stored(self, store: UserStore, tmp_path: Path) -> None:
        """The database file never contains the password."""
        store.create("Jane", "jane@example.com", PASSWORD)
        assert PASSWORD.encode() not in (tmp_path / "users.db").read_bytes()

    def test_settings_round_trip(self, store: UserStore) -> None:
        """Defaults before saving; saved values come back."""
        store.create("Jane", "jane@example.com", PASSWORD)
        assert store.settings("jane@example.com") == UserSettings()
        saved = UserSettings(default_goal="revenue", annual_weeks=50, show_explanations=False)
        store.save_settings("jane@example.com", saved)
        assert store.settings("jane@example.com") == saved


class TestUserSettings:
    """Stored settings tolerate unknown and wrong keys."""

    def test_from_dict(self) -> None:
        """Unknown keys are ignored."""
        settings = UserSettings.from_dict({"annual_weeks": 50, "removed_option": 1})
        assert settings.annual_weeks == 50
