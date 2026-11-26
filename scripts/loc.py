"""Lines-of-code report: Python via gnost, notebooks cell by cell, plus GitHub language share.

Python LOC comes from `uvx gnost summary` (gnost does not read notebooks). Notebook
code cells are counted like gnost counts .py files (code / comments / blanks). The
language share mirrors GitHub Linguist: bytes of git-tracked files per language.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
console = Console()


@dataclass(frozen=True)
class NotebookStats:
    """Line counts for one notebook; code-cell lines split like gnost does for .py files."""

    path: str
    code_cells: int
    code: int
    comments: int
    blanks: int
    markdown: int

    @property
    def loc(self) -> int:
        """Code-cell lines including comments and blanks (same definition as gnost LOC)."""
        return self.code + self.comments + self.blanks


class GitFiles:
    """Lists git-tracked files and their Linguist attributes.

    Tracked files only, like GitHub: ignored data and caches never count.
    """

    LINGUIST_ATTRS: ClassVar[tuple[str, ...]] = (
        "linguist-vendored", "linguist-generated", "linguist-documentation",
    )  # fmt: skip

    def __init__(self, root: Path) -> None:
        self._root = root

    def tracked(self, pattern: str = "") -> list[str]:
        """Tracked paths, optionally filtered by a git pathspec such as '*.ipynb'."""
        args = ["git", "ls-files", "-z", *([pattern] if pattern else [])]
        out = subprocess.run(args, cwd=self._root, capture_output=True, check=True).stdout
        return [p for p in out.decode().split("\0") if p]

    def excluded_by_attributes(self, paths: Sequence[str]) -> set[str]:
        """Paths marked vendored, generated or documentation in .gitattributes."""
        if not paths:
            return set()
        args = ["git", "check-attr", "-z", "--stdin", *self.LINGUIST_ATTRS]
        stdin = ("\0".join(paths) + "\0").encode()
        result = subprocess.run(args, cwd=self._root, input=stdin, capture_output=True, check=True)
        out = result.stdout.decode().split("\0")  # path, attribute, value triples
        triples = zip(out[0::3], out[1::3], out[2::3], strict=False)
        return {path for path, _, value in triples if value in {"set", "true"}}


class NotebookCounter:
    """Counts lines in a notebook's code and markdown cells.

    Comment = line starting with '#'; IPython magics ('%', '!') count as code.
    """

    def count(self, root: Path, relative: str) -> NotebookStats:
        """Read one .ipynb and return its line counts."""
        cells = json.loads((root / relative).read_text(encoding="utf-8")).get("cells", [])
        code_lines = [line for c in cells if c["cell_type"] == "code" for line in self._lines(c)]
        markdown = sum(len(self._lines(c)) for c in cells if c["cell_type"] == "markdown")
        stripped = [line.strip() for line in code_lines]
        return NotebookStats(
            path=relative,
            code_cells=sum(c["cell_type"] == "code" for c in cells),
            code=sum(bool(s) and not s.startswith("#") for s in stripped),
            comments=sum(s.startswith("#") for s in stripped),
            blanks=sum(not s for s in stripped),
            markdown=markdown,
        )

    @staticmethod
    def _lines(cell: dict) -> list[str]:
        """Source lines of a cell (source may be a list or a single string)."""
        source = cell.get("source", "")
        text = "".join(source) if isinstance(source, list) else source
        return text.splitlines()


class LinguistShare:
    """Approximates GitHub's language bar: bytes of tracked files per programming language.

    Like Linguist, it skips docs/, prose (Markdown), data (YAML/TOML/JSON) and attribute exclusions.
    """

    LANGUAGES: ClassVar[dict[str, str]] = {
        ".py": "Python", ".ipynb": "Jupyter Notebook", ".sh": "Shell", ".ps1": "PowerShell",
        ".js": "JavaScript", ".ts": "TypeScript", ".html": "HTML", ".css": "CSS", ".sql": "SQL",
    }  # fmt: skip
    FILENAMES: ClassVar[dict[str, str]] = {"Makefile": "Makefile", "Dockerfile": "Dockerfile"}
    DOC_DIRS: ClassVar[tuple[str, ...]] = ("docs/", "doc/", "documentation/")

    def __init__(self, files: GitFiles, root: Path) -> None:
        self._files = files
        self._root = root

    def shares(self) -> dict[str, int]:
        """Bytes per language, largest first."""
        paths = [p for p in self._files.tracked() if not p.startswith(self.DOC_DIRS)]
        excluded = self._files.excluded_by_attributes(paths)
        totals: dict[str, int] = {}
        for path in paths:
            language = self._language(path)
            if language and path not in excluded:
                totals[language] = totals.get(language, 0) + (self._root / path).stat().st_size
        return dict(sorted(totals.items(), key=lambda kv: -kv[1]))

    def _language(self, path: str) -> str | None:
        """Language of a path by file name, then extension."""
        name = Path(path).name
        return self.FILENAMES.get(name) or self.LANGUAGES.get(Path(path).suffix.lower())


class LocReport:
    """Prints the gnost Python summary, notebook counts and the language share.

    Each part is independent, so a missing gnost does not block the notebook count.
    """

    def __init__(self, root: Path, run_gnost: bool = True) -> None:
        self._root = root
        self._run_gnost = run_gnost
        self._files = GitFiles(root)

    def render(self) -> int:
        """Print all sections; returns a process exit code."""
        if self._run_gnost:
            self._gnost()
        notebooks = self._files.tracked("*.ipynb")
        skipped = self._files.excluded_by_attributes(notebooks)  # e.g. archive/ (vendored)
        counter = NotebookCounter()
        stats = [counter.count(self._root, p) for p in notebooks if p not in skipped]
        console.print(self._notebook_table(stats))
        console.print(self._share_table(LinguistShare(self._files, self._root).shares()))
        return 0

    def _gnost(self) -> None:
        """Run `uvx gnost summary` for Python and pass its output through."""
        try:
            subprocess.run(["uvx", "gnost", "summary"], cwd=self._root, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as err:
            console.print(f"[yellow]gnost unavailable ({err}); skipping Python summary[/yellow]")

    @staticmethod
    def _notebook_table(stats: list[NotebookStats]) -> Table:
        """Per-notebook rows plus a total row."""
        table = Table(title="Notebooks (code cells, counted like gnost; vendored skipped)")
        for column in ("Notebook", "Cells", "Code", "Comments", "Blanks", "LOC", "Markdown"):
            table.add_column(column, justify="left" if column == "Notebook" else "right")
        for s in stats:
            table.add_row(
                s.path, *map(str, (s.code_cells, s.code, s.comments, s.blanks, s.loc, s.markdown))
            )
        fields = ("code_cells", "code", "comments", "blanks", "loc", "markdown")
        totals = [sum(getattr(s, f) for s in stats) for f in fields]
        table.add_row(f"[bold]Total ({len(stats)})[/bold]", *[f"[bold]{v}[/bold]" for v in totals])
        return table

    @staticmethod
    def _share_table(shares: dict[str, int]) -> Table:
        """Language share by bytes, as GitHub's language bar shows it."""
        total = sum(shares.values()) or 1
        table = Table(title="GitHub language bar (Linguist-style, bytes of tracked files)")
        table.add_column("Language")
        table.add_column("KB", justify="right")
        table.add_column("Share", justify="right")
        for language, size in shares.items():
            table.add_row(language, f"{size / 1024:.1f}", f"{size / total:.1%}")
        return table


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Lines-of-code report for Python and notebooks")
    cli.add_argument("--no-gnost", action="store_true", help="skip the gnost Python summary")
    sys.exit(LocReport(ROOT, run_gnost=not cli.parse_args().no_gnost).render())
