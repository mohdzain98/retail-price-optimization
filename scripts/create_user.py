"""Create a local account for the Streamlit simulator (run with `make user`).

Prompts for name, email and password (hidden, typed twice), re-asks on invalid
input, refuses an email that already has an account, and stores a salted scrypt
hash in the local SQLite file (data/users.db, or RPO_USERS_DB). Exit code 1 on abort.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from getpass import getpass

from src.simulator.accounts import UserStore
from src.utils import AccountError

MAX_TRIES = 3


class CreateUserCli:
    """Interactive account creation; prompts are injectable so tests can drive it.

    Each field gets MAX_TRIES attempts before the command gives up.
    """

    def __init__(
        self,
        store: UserStore,
        ask: Callable[[str], str] = input,
        ask_secret: Callable[[str], str] = getpass,
        say: Callable[[str], None] = print,
    ) -> None:
        self._store = store
        self._ask = ask
        self._ask_secret = ask_secret
        self._say = say

    def run(self) -> int:
        """Prompt, validate and create the account; returns a process exit code."""
        self._say("Create a simulator account (stored locally, password hashed).")
        try:
            name = self._retry(lambda: self._store.validator.name(self._ask("Name: ")))
            email = self._retry(self._new_email)
            password = self._retry(self._password)
            user = self._store.create(name, email, password)
        except AccountError as err:
            self._say(f"Account not created: {err}")
            return 1
        except (KeyboardInterrupt, EOFError):
            self._say("\nCancelled.")
            return 1
        self._say(f"Created account for {user.name} <{user.email}>. Start the app with `make app`.")
        return 0

    def _new_email(self) -> str:
        """A valid email without an existing account."""
        email = self._store.validator.email(self._ask("Email: "))
        if self._store.exists(email):
            raise AccountError(f"An account for {email} already exists")
        return email

    def _password(self) -> str:
        """A strong password, typed twice."""
        password = self._store.validator.password(self._ask_secret("Password: "))
        if self._ask_secret("Repeat password: ") != password:
            raise AccountError("Passwords do not match")
        return password

    def _retry(self, step: Callable[[], str]) -> str:
        """Run a prompt step up to MAX_TRIES times, showing each error."""
        for attempt in range(1, MAX_TRIES + 1):
            try:
                return step()
            except AccountError as err:
                if attempt == MAX_TRIES:
                    raise
                self._say(f"  {err}. Try again.")
        raise AccountError("Too many attempts")  # unreachable; keeps type checkers happy


if __name__ == "__main__":
    sys.exit(CreateUserCli(UserStore(UserStore.default_path())).run())
