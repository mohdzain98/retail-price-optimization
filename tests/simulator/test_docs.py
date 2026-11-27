"""Tests for src.simulator.docs: docs adapted for display in the app.

Checks the math-fence conversion and that the real docs load. One test class per
production class.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.simulator.docs import DocReader
from src.utils import MissingFileError


class TestDocReader:
    """DocReader converts ```math fences and leaves other blocks alone."""

    def test_math_fence(self, tmp_path: Path) -> None:
        """A math fence becomes a $$ block; Mermaid stays."""
        (tmp_path / "doc.md").write_text(
            "Text \\$5\n```math\nx^2\n```\n```mermaid\nflowchart TD\n```\n"
        )
        text = DocReader(tmp_path).read("doc")
        assert "$$\nx^2\n$$" in text and "```math" not in text
        assert "```mermaid" in text and "\\$5" in text

    def test_real_docs(self) -> None:
        """Methodology and architecture load without math fences."""
        reader = DocReader()
        for name in ("methodology", "architecture"):
            assert "```math" not in reader.read(name)

    def test_missing(self, tmp_path: Path) -> None:
        """A missing doc raises MissingFileError."""
        with pytest.raises(MissingFileError):
            DocReader(tmp_path).read("nope")
