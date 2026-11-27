"""Fail a pull request that adds data, secrets, local notes or oversized files.

Lists files added or modified relative to a base branch (`git diff base...HEAD`) and
checks them against forbidden patterns: the licensed dataset and pipeline outputs,
account databases, secrets, local plans and caches, and files above a size limit.
Run in CI on every pull request and locally with `make pr-check`. Exit code 1 on a hit.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

MAX_MB = 2.0
ALLOWED: frozenset[str] = frozenset(
    {
        ".env.example",  # template without secrets
        "data/raw/.gitkeep",
        "data/processed/.gitkeep",
        "archive/retail_price.csv",  # sample data of the original 2022 notebook
    }
)
FORBIDDEN: tuple[tuple[str, str], ...] = (
    ("data/*", "data folder (licensed dataset, pipeline outputs, accounts)"),
    ("*.parquet", "data file"),
    ("*.pkl", "pickled model or data"),
    ("*.pickle", "pickled model or data"),
    ("*.csv", "data file"),
    ("*.xlsx", "data file (the dunnhumby workbook is licensed)"),
    ("*.xls", "data file"),
    ("*.zip", "archive"),
    ("*.feather", "data file"),
    ("*.h5", "data file"),
    ("*.db", "database (may hold accounts)"),
    ("*.db-journal", "database journal"),
    ("*.sqlite", "database"),
    ("*.sqlite3", "database"),
    (".env", "secrets"),
    (".env.*", "secrets"),
    ("*/.env", "secrets"),
    (".streamlit/secrets.toml", "secrets"),
    ("docs/plans/*", "local planning notes"),
    (".cache/*", "local cache"),
    ("*.ipynb_checkpoints/*", "notebook checkpoints"),
    ("*__pycache__/*", "compiled Python"),
)


@dataclass(frozen=True)
class Violation:
    """One file that must not be in the pull request, and why."""

    path: str
    reason: str


class PrFileGuard:
    """Checks changed paths against forbidden patterns and a size limit.

    Paths in ALLOWED always pass; `size_of` returns a file's size in bytes (None if
    the file no longer exists, e.g. a rename source).
    """

    def __init__(self, max_bytes: int = int(MAX_MB * 1024 * 1024)) -> None:
        self._max_bytes = max_bytes

    def violations(
        self, paths: Iterable[str], size_of: Callable[[str], int | None]
    ) -> list[Violation]:
        """Every forbidden or oversized path, in the given order."""
        found = []
        for path in paths:
            if path in ALLOWED:
                continue
            reason = next((why for pattern, why in FORBIDDEN if fnmatch(path, pattern)), None)
            size = size_of(path)
            if reason is None and size is not None and size > self._max_bytes:
                limit = self._max_bytes / 1024 / 1024
                reason = f"{size / 1024 / 1024:.1f} MB, above the {limit:.0f} MB limit"
            if reason:
                found.append(Violation(path, reason))
        return found


def changed_files(base: str) -> list[str]:
    """Paths added, copied, modified or renamed on this branch since it left `base`."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base}...HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def file_size(path: str) -> int | None:
    """Size of a file in the working tree, or None if it does not exist."""
    target = Path(path)
    return target.stat().st_size if target.is_file() else None


def main(argv: Sequence[str] | None = None) -> int:
    """Check the branch against `--base`; print every violation; exit 1 if any."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main", help="branch the PR merges into")
    args = parser.parse_args(argv)
    paths = changed_files(args.base)
    found = PrFileGuard().violations(paths, file_size)
    if not found:
        print(f"OK: {len(paths)} changed files, no data, secrets or oversized files.")
        return 0
    print("These files must not be committed:")
    for violation in found:
        print(f"  {violation.path}: {violation.reason}")
    print("Remove them from the branch (git rm --cached <file>) and check .gitignore.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
