"""Tests for src.pipeline: stage registry and CLI behaviour.

The CLI must reject unknown stages and turn project errors (such as a missing
raw file) into exit code 1 instead of a traceback.
One test class per production class.
"""

from __future__ import annotations

import pytest

from src.pipeline import STAGES, PipelineCli, PrepStage, StageName
from src.utils import MissingFileError


class TestPrepStage:
    """PrepStage is registered under its stage name."""

    def test_registered(self) -> None:
        """Registry resolves 'prep' to PrepStage."""
        assert STAGES.get(StageName.PREP) is PrepStage


class TestPipelineCli:
    """PipelineCli parses arguments and maps errors to exit codes."""

    def test_unknown_stage_exits(self) -> None:
        """argparse rejects stages that are not registered."""
        with pytest.raises(SystemExit):
            PipelineCli().run(["not_a_stage"])

    def test_project_error_returns_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An RPOError inside a stage becomes exit code 1."""

        def boom(self: PrepStage) -> None:
            raise MissingFileError("no data")

        monkeypatch.setattr(PrepStage, "run", boom)
        assert PipelineCli().run(["prep"]) == 1
