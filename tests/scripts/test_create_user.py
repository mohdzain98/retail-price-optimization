"""Tests for scripts/create_user.py: the interactive `make user` command.

Prompts are replaced by scripted answers; each test uses its own SQLite file.
One test class per production class.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from scripts.create_user import CreateUserCli
from src.simulator.accounts import UserStore


def _answers(*values: str) -> Callable[[str], str]:
    """A prompt function returning the given answers in order."""
    queue = list(values)
    return lambda _prompt: queue.pop(0)


def _cli(store: UserStore, plain: list[str], secret: list[str], out: list[str]) -> CreateUserCli:
    """CLI with scripted input and captured output."""
    return CreateUserCli(store, _answers(*plain), _answers(*secret), out.append)


class TestCreateUserCli:
    """The command validates, retries and refuses duplicates."""

    def test_creates_account(self, tmp_path: Path) -> None:
        """Valid answers create the account and exit 0."""
        store, out = UserStore(tmp_path / "u.db"), list[str]()
        cli = _cli(store, ["Jane Doe", "jane@gmail.com"], ["secret123", "secret123"], out)
        assert cli.run() == 0
        assert store.authenticate("jane@gmail.com", "secret123") is not None

    def test_retries_bad_input(self, tmp_path: Path) -> None:
        """Bad email and mismatched password are re-asked, then accepted."""
        store, out = UserStore(tmp_path / "u.db"), list[str]()
        cli = _cli(
            store,
            ["Jane", "not-an-email", "jane@gmail.com"],
            ["secret123", "different1", "secret123", "secret123"],
            out,
        )
        assert cli.run() == 0
        assert any("valid email" in line for line in out)
        assert any("do not match" in line for line in out)

    def test_existing_email(self, tmp_path: Path) -> None:
        """An email with an account is refused three times, then the command exits 1."""
        store, out = UserStore(tmp_path / "u.db"), list[str]()
        store.create("Jane", "jane@gmail.com", "secret123")
        cli = _cli(store, ["Other", *["JANE@gmail.com"] * 3], [], out)
        assert cli.run() == 1
        assert store.count() == 1
        assert any("already exists" in line for line in out)

    def test_cancel(self, tmp_path: Path) -> None:
        """Ctrl-C exits 1 without creating anything."""

        def interrupt(_prompt: str) -> str:
            raise KeyboardInterrupt

        store, out = UserStore(tmp_path / "u.db"), list[str]()
        assert CreateUserCli(store, interrupt, interrupt, out.append).run() == 1
        assert store.count() == 0
