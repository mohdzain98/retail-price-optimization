"""Project docs (methodology, architecture) prepared for display inside the app.

GitHub renders ```math fences, Streamlit renders $$ blocks; DocReader converts one
to the other and leaves Mermaid fences, tables and escaped dollars untouched.
Used by the app's Glossary dialog.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from src.config import DOCS
from src.utils import MissingFileError


class DocReader:
    """Reads a Markdown doc from docs/ and adapts its math blocks for st.markdown.

    Raises MissingFileError when the doc does not exist.
    """

    MATH_FENCE: ClassVar[re.Pattern[str]] = re.compile(r"```math\n(.*?)\n```", re.DOTALL)

    def __init__(self, directory: Path = DOCS) -> None:
        self._directory = directory

    def read(self, name: str) -> str:
        """Contents of docs/<name>.md with ```math fences turned into $$ blocks."""
        path = self._directory / f"{name}.md"
        if not path.exists():
            raise MissingFileError(f"{path} not found")
        return self.MATH_FENCE.sub(lambda m: f"$$\n{m.group(1)}\n$$", path.read_text())
