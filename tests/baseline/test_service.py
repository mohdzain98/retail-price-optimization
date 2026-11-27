"""Tests for src.baseline.service: the facade runs end to end and writes outputs.

Uses the stub model through the factory, so the whole chain (effects, rules,
decomposition, summary) runs on the 6-week panel. One test class per production class.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline import BaselineFactory, BaselineOutputs, BaselineService
from src.config import Config, ModelName
from src.schema import LIFT_COLS, Col
from tests.baseline.conftest import LinearStub


def _outputs(tmp_path: Path) -> BaselineOutputs:
    """Output paths under tmp_path."""
    return BaselineOutputs(tmp_path / "weekly.parquet", tmp_path / "summary.parquet")


class TestBaselineService:
    """BaselineService decomposes and writes both tables."""

    def test_run(self, week_panel: pd.DataFrame, final_table: pd.DataFrame, tmp_path: Path) -> None:
        """Both files written; lifts sum to incremental; one summary row per product."""
        models = {ModelName.OLS_PER_PRODUCT: LinearStub()}
        outputs = _outputs(tmp_path)
        report = (
            BaselineFactory(Config.load(), models, outputs).build().run(week_panel, final_table)
        )
        assert outputs.weekly.exists() and outputs.summary.exists()
        weekly = report.weekly
        lifts = weekly[list(LIFT_COLS.values())].sum(axis=1)
        np.testing.assert_allclose(lifts, weekly[Col.INCREMENTAL])
        assert len(report.summary) == week_panel[Col.UPC].nunique()
        assert (weekly[Col.BASELINE] > 0).all()


class TestBaselineFactory:
    """BaselineFactory wires a ready-to-run service."""

    def test_builds_service(self, tmp_path: Path) -> None:
        """Returns a BaselineService."""
        models = {ModelName.OLS_PER_PRODUCT: LinearStub()}
        service = BaselineFactory(Config.load(), models, _outputs(tmp_path)).build()
        assert isinstance(service, BaselineService)
