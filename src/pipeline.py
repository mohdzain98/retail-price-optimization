"""Command-line entry point and composition root for pipeline stages.

Usage: python -m src.pipeline {prep,elasticity} [--no-cache]
Each Stage registers in STAGES and wires its layer's facade from config and
paths; PipelineCli parses arguments and runs the chosen stage.
"""

from __future__ import annotations

import argparse
import logging
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

import pandas as pd

from src.config import DATA_PROCESSED, PANEL_PATH, Config
from src.elasticity import ElasticityFactory, ElasticityOutputs
from src.prep import DataPrepFactory, LoaderSettings
from src.utils import AppLogger, MissingFileError, Registry, RPOError

logger = AppLogger.get_logger(__name__)


class StageName(StrEnum):
    """Pipeline stages runnable from the CLI."""

    PREP = "prep"
    ELASTICITY = "elasticity"


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


@STAGES.register
class ElasticityStage(Stage):
    """Phase 2: panel.parquet -> elasticities (final, all models, comparison)."""

    name = StageName.ELASTICITY

    def run(self) -> None:
        """Fit configured models on the prepared panel."""
        if not PANEL_PATH.exists():
            raise MissingFileError(f"{PANEL_PATH} not found; run `make prep` first")
        service = ElasticityFactory(self._cfg, ElasticityOutputs.default()).build()
        service.run(pd.read_parquet(PANEL_PATH))


class PipelineCli:
    """Parses CLI arguments and runs one registered stage.

    Converts project errors into a logged message and exit code 1.
    """

    OWN_HANDLER_LOGGERS: tuple[str, ...] = ("pymc",)  # already print via their own handler

    def run(self, argv: Sequence[str] | None = None) -> int:
        """Run the stage named in argv; returns a process exit code."""
        args = self._parser().parse_args(argv)
        AppLogger.init()
        for name in self.OWN_HANDLER_LOGGERS:
            logging.getLogger(name).propagate = False
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
