"""Tests for scripts/check_pr_files.py: the pull-request guard against data and secrets.

Pattern checks use an injected size function; the end-to-end test builds a
throwaway git repository with a branch that adds a forbidden file.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.check_pr_files import ALLOWED, PrFileGuard, changed_files, main

MB = 1024 * 1024


def _size(path: str) -> int | None:
    """Every file is small."""
    return 100


class TestPrFileGuard:
    """Forbidden patterns, the allowlist and the size limit."""

    @pytest.mark.parametrize(
        "path",
        [
            "data/raw/dunnhumby - Breakfast at the Frat.xlsx", "data/processed/panel.parquet",
            "data/users.db", "results.csv", "models/elasticity_models.pkl", "app/users.sqlite3",
            ".env", ".env.local", ".streamlit/secrets.toml", "docs/plans/phase5.md",
            ".cache/research/notes.md", "src/__pycache__/x.pyc",
        ],
    )  # fmt: skip
    def test_forbidden(self, path: str) -> None:
        """Data, databases, secrets, notes and caches are rejected."""
        assert [v.path for v in PrFileGuard().violations([path], _size)] == [path]

    @pytest.mark.parametrize("path", sorted(ALLOWED) + ["src/pipeline.py", "README.md"])
    def test_allowed(self, path: str) -> None:
        """Code, docs and the explicit allowlist pass."""
        assert PrFileGuard().violations([path], _size) == []

    def test_size_limit(self) -> None:
        """A large figure fails; a deleted file (no size) passes."""
        guard = PrFileGuard(max_bytes=2 * MB)
        sizes = {"reports/figures/big.png": 3 * MB, "reports/figures/ok.png": MB}
        found = guard.violations([*sizes, "gone.py"], sizes.get)
        assert [v.path for v in found] == ["reports/figures/big.png"]
        assert "MB limit" in found[0].reason


class TestMain:
    """End to end in a temporary git repository."""

    def _git(self, repo: Path, *args: str) -> None:
        """Run git quietly in the repository."""
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    def test_branch_with_data_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A branch adding a parquet file exits 1; one adding only code exits 0."""
        repo = tmp_path / "repo"
        repo.mkdir()
        self._git(repo, "init", "-q", "-b", "main")
        self._git(repo, "config", "user.email", "ci@example.com")
        self._git(repo, "config", "user.name", "CI")
        (repo / "README.md").write_text("x")
        self._git(repo, "add", ".")
        self._git(repo, "commit", "-qm", "base")
        self._git(repo, "checkout", "-qb", "feature")
        (repo / "code.py").write_text("x = 1\n")
        self._git(repo, "add", ".")
        self._git(repo, "commit", "-qm", "code")
        monkeypatch.chdir(repo)
        assert changed_files("main") == ["code.py"]
        assert main(["--base", "main"]) == 0
        (repo / "data").mkdir()
        (repo / "data" / "panel.parquet").write_bytes(b"x")
        self._git(repo, "add", ".")
        self._git(repo, "commit", "-qm", "data")
        assert main(["--base", "main"]) == 1
        assert "data/panel.parquet" in capsys.readouterr().out
