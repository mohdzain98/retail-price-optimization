"""Local user accounts for the simulator: SQLite store, scrypt hashes and user settings.

For a local, single-machine app: accounts are created in a terminal (`make user`)
and checked at sign-in. Passwords are stored only as salted scrypt hashes. Each
user's app settings are saved next to the account. The path can be overridden.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar

from src.config import USERS_DB_PATH, Objective
from src.utils import AccountError

USERS_DB_ENV = "RPO_USERS_DB"


@dataclass(frozen=True)
class User:
    """A signed-in user (never carries the password hash)."""

    name: str
    email: str


@dataclass(frozen=True)
class UserSettings:
    """App preferences saved per user and applied after sign-in.

    Unknown or invalid stored values fall back to the defaults.
    """

    default_goal: str = Objective.PROFIT.value
    annual_weeks: int = 52
    frontier_points: int = 16
    min_marginal_return: float = 0.25
    show_explanations: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> UserSettings:
        """Settings from stored JSON, ignoring keys this version does not know."""
        known = {f.name for f in fields(cls)}
        try:
            return cls(**{k: v for k, v in raw.items() if k in known})
        except TypeError:
            return cls()


class PasswordHasher:
    """Salted scrypt password hashes in the form scrypt$n$r$p$salt$hash (hex).

    Verification recomputes the hash and compares in constant time.
    """

    N: ClassVar[int] = 2**14
    R: ClassVar[int] = 8
    P: ClassVar[int] = 1
    SALT_BYTES: ClassVar[int] = 16

    def hash(self, password: str) -> str:
        """New hash with a random salt."""
        salt = secrets.token_bytes(self.SALT_BYTES)
        digest = self._scrypt(password, salt, self.N, self.R, self.P)
        return f"scrypt${self.N}${self.R}${self.P}${salt.hex()}${digest.hex()}"

    def verify(self, password: str, stored: str) -> bool:
        """True when `password` matches the stored hash; False for malformed hashes."""
        try:
            scheme, n, r, p, salt, digest = stored.split("$")
            if scheme != "scrypt":
                return False
            computed = self._scrypt(password, bytes.fromhex(salt), int(n), int(r), int(p))
        except ValueError:
            return False
        return hmac.compare_digest(computed, bytes.fromhex(digest))

    @staticmethod
    def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
        """Raw scrypt digest."""
        return hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, dklen=32)


class AccountValidator:
    """Rules for names, emails and passwords; raises AccountError with a readable reason.

    Passwords need 8+ characters with at least one letter and one digit.
    """

    EMAIL: ClassVar[re.Pattern[str]] = re.compile(
        r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    )
    MIN_PASSWORD: ClassVar[int] = 8
    MAX_NAME: ClassVar[int] = 80

    def name(self, value: str) -> str:
        """Trimmed display name."""
        name = " ".join(value.split())
        if not name or len(name) > self.MAX_NAME:
            raise AccountError(f"Name must be 1-{self.MAX_NAME} characters")
        return name

    def email(self, value: str) -> str:
        """Lower-cased, trimmed email."""
        email = value.strip().lower()
        if not self.EMAIL.match(email):
            raise AccountError("Enter a valid email address, e.g. name@gmail.com")
        return email

    def password(self, value: str) -> str:
        """The password, if strong enough."""
        if len(value) < self.MIN_PASSWORD:
            raise AccountError(f"Password needs at least {self.MIN_PASSWORD} characters")
        if not (re.search(r"[A-Za-z]", value) and re.search(r"\d", value)):
            raise AccountError("Password needs at least one letter and one digit")
        return value


class UserStore:
    """SQLite table of accounts (name, email, password hash, settings JSON).

    Opens a short-lived connection per call, so it is safe across Streamlit threads.
    """

    SCHEMA: ClassVar[str] = (
        "CREATE TABLE IF NOT EXISTS users ("
        " email TEXT PRIMARY KEY, name TEXT NOT NULL, password_hash TEXT NOT NULL,"
        " settings TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )

    def __init__(
        self,
        path: Path,
        hasher: PasswordHasher | None = None,
        validator: AccountValidator | None = None,
    ) -> None:
        self._path = path
        self._hasher = hasher or PasswordHasher()
        self._validator = validator or AccountValidator()

    @staticmethod
    def default_path() -> Path:
        """RPO_USERS_DB if set, else data/users.db."""
        return Path(os.environ.get(USERS_DB_ENV, USERS_DB_PATH))

    @property
    def validator(self) -> AccountValidator:
        """The input rules this store enforces."""
        return self._validator

    def exists(self, email: str) -> bool:
        """Whether an account uses this email (case-insensitive)."""
        with self._connect() as db:
            row = db.execute(
                "SELECT 1 FROM users WHERE email = ?", (email.strip().lower(),)
            ).fetchone()
        return row is not None

    def count(self) -> int:
        """Number of accounts."""
        with self._connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def create(self, name: str, email: str, password: str) -> User:
        """Validate and store a new account; raises AccountError if invalid or taken."""
        user = User(self._validator.name(name), self._validator.email(email))
        hashed = self._hasher.hash(self._validator.password(password))
        try:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO users (email, name, password_hash) VALUES (?, ?, ?)",
                    (user.email, user.name, hashed),
                )
        except sqlite3.IntegrityError as err:
            raise AccountError(f"An account for {user.email} already exists") from err
        return user

    def authenticate(self, email: str, password: str) -> User | None:
        """The user if the email exists and the password matches, else None."""
        with self._connect() as db:
            row = db.execute(
                "SELECT name, email, password_hash FROM users WHERE email = ?",
                (email.strip().lower(),),
            ).fetchone()
        if row is None:
            self._hasher.hash(password)  # similar timing for unknown emails
            return None
        name, stored_email, hashed = row
        return User(name, stored_email) if self._hasher.verify(password, hashed) else None

    def settings(self, email: str) -> UserSettings:
        """Saved settings of a user (defaults if none or unreadable)."""
        with self._connect() as db:
            row = db.execute("SELECT settings FROM users WHERE email = ?", (email,)).fetchone()
        try:
            return UserSettings.from_dict(json.loads(row[0])) if row else UserSettings()
        except json.JSONDecodeError:
            return UserSettings()

    def save_settings(self, email: str, settings: UserSettings) -> None:
        """Persist a user's settings."""
        with self._connect() as db:
            db.execute(
                "UPDATE users SET settings = ? WHERE email = ?",
                (json.dumps(asdict(settings)), email),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Connection with the schema in place; commits on success, always closes."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self._path)
        try:
            db.execute(self.SCHEMA)
            yield db
            db.commit()
        finally:
            db.close()
