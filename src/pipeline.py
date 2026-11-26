"""Command-line entry point and composition root for pipeline stages.

Usage: python -m src.pipeline prep [--no-cache]
Each Stage registers in STAGES and wires its layer's facade from config and
paths; PipelineCli parses arguments and runs the chosen stage.
"""

from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from src.config import DATA_PROCESSED, PANEL_PATH, Config
from src.prep import DataPrepFactory, LoaderSettings
from src.utils import AppLogger, Registry, RPOError

logger = AppLogger.get_logger(__name__)


class StageName(StrEnum):
    """Pipeline stages runnable from the CLI."""

    PREP = "prep"


@dataclass(frozen=True)
class RunOptions:
    """CLI options shared by all stages."""

    use_cache: bool = True


class Stage(ABC):
    """One runnable pipeline stage; subclasses register in STAGES.

    Receives config and options; builds its own facade (composition root).
    """

    name: ClassVar[StageName]

    def __init__(self, cfg: Config, options: RunOptions) -> None:
        self._cfg = cfg
        self._options = options

    @abstractmethod
    def run(self) -> None:
        """Execute the stage and write its outputs."""


STAGES: Registry[StageName, type[Stage]] = Registry("name")


@STAGES.register
class PrepStage(Stage):
    """Phase 1: raw workbook -> data/processed/panel.parquet."""

    name = StageName.PREP

    def run(self) -> None:
        """Build and run the data-prep service."""
        settings = LoaderSettings.from_config(self._cfg, cache_dir=DATA_PROCESSED)
        DataPrepFactory(self._cfg, settings, PANEL_PATH).build().run(self._options.use_cache)


class PipelineCli:
    """Parses CLI arguments and runs one registered stage.

    Converts project errors into a logged message and exit code 1.
    """

    def run(self, argv: Sequence[str] | None = None) -> int:
        """Run the stage named in argv; returns a process exit code."""
        args = self._parser().parse_args(argv)
        AppLogger.init()
        try:
            stage_cls = STAGES.get(StageName(args.stage))
            stage_cls(Config.load(), RunOptions(use_cache=not args.no_cache)).run()
        except RPOError as err:
            logger.error(str(err), extra={"error": type(err).__name__})
            return 1
        return 0

    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        """Argument parser with one positional stage name."""
        parser = argparse.ArgumentParser(description="Retail price optimization pipeline")
        parser.add_argument("stage", choices=[stage.name.value for stage in STAGES])
        parser.add_argument("--no-cache", action="store_true", help="re-read the raw Excel file")
        return parser


if __name__ == "__main__":
    sys.exit(PipelineCli().run())
