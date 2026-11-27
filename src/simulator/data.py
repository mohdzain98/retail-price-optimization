"""Processed pipeline outputs the simulator reads, loaded once per data directory.

SimulatorData holds the Phase 4 and Phase 5 tables; `load` checks every file exists
and names the `make` target that produces a missing one. The directory defaults to
data/processed and can be overridden with RPO_DATA_DIR (used by app tests).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd

from src.config import (
    BUDGET_FRONTIER_PATH,
    DATA_PROCESSED,
    FORECAST_BACKTEST_PATH,
    FORECAST_PATH,
    OPTIMIZATION_SUMMARY_PATH,
    PROMO_CANDIDATES_PATH,
    PROMO_PLAN_PATH,
    PROMO_SUMMARY_PATH,
    RECOMMENDATIONS_PATH,
)
from src.schema import Col
from src.utils import MissingFileError

DATA_DIR_ENV = "RPO_DATA_DIR"
SOURCES: dict[str, tuple[Path, str]] = {
    "recommendations": (RECOMMENDATIONS_PATH, "optimize"),
    "optimization_summary": (OPTIMIZATION_SUMMARY_PATH, "optimize"),
    "forecast": (FORECAST_PATH, "promo"),
    "backtest": (FORECAST_BACKTEST_PATH, "promo"),
    "candidates": (PROMO_CANDIDATES_PATH, "promo"),
    "plans": (PROMO_PLAN_PATH, "promo"),
    "frontier": (BUDGET_FRONTIER_PATH, "promo"),
    "promo_summary": (PROMO_SUMMARY_PATH, "promo"),
}


@dataclass(frozen=True)
class SimulatorData:
    """Every table the simulator needs: prices (Phase 4) and forecast/promo (Phase 5).

    Field names match SOURCES; files are read by name from one directory.
    """

    recommendations: pd.DataFrame
    optimization_summary: pd.DataFrame
    forecast: pd.DataFrame
    backtest: pd.DataFrame
    candidates: pd.DataFrame
    plans: pd.DataFrame
    frontier: pd.DataFrame
    promo_summary: pd.DataFrame

    @staticmethod
    def directory() -> Path:
        """RPO_DATA_DIR if set, else data/processed."""
        return Path(os.environ.get(DATA_DIR_ENV, DATA_PROCESSED))

    @classmethod
    def load(cls, directory: Path | None = None) -> SimulatorData:
        """Read every table; raises MissingFileError naming the make targets to run."""
        root = directory or cls.directory()
        missing = {target for path, target in SOURCES.values() if not (root / path.name).exists()}
        if missing:
            steps = ", ".join(f"`make {t}`" for t in ("optimize", "promo") if t in missing)
            raise MissingFileError(f"Pipeline outputs missing in {root}; run {steps} first")
        tables = {f.name: pd.read_parquet(root / SOURCES[f.name][0].name) for f in fields(cls)}
        return cls(**tables)

    @property
    def objectives(self) -> list[str]:
        """Objectives solved by the pipeline, in their stored order."""
        return list(dict.fromkeys(self.recommendations[Col.OBJECTIVE]))
