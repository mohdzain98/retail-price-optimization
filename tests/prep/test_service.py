"""Tests for src.prep.service: the DataPrepService facade and its factory.

Runs the full Phase 1 chain on the synthetic workbook and checks the written
parquet matches the returned panel and contains every expected column group.
One test class per production class.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config
from src.prep import DataPrepFactory, DataPrepService, LoaderSettings
from src.schema import Col, Split


class TestDataPrepService:
    """DataPrepService runs the chain and writes the panel."""

    def test_writes_panel(
        self, cfg: Config, loader_settings: LoaderSettings, tmp_path: Path
    ) -> None:
        """Parquet output equals the returned panel and has all column groups."""
        out_path = tmp_path / "out" / "panel.parquet"
        panel = DataPrepFactory(cfg, loader_settings, out_path).build().run()
        pd.testing.assert_frame_equal(pd.read_parquet(out_path), panel, check_dtype=False)
        expected = {Col.ELIGIBLE, Col.LN_PRICE, Col.HOLIDAY_WEEK, Col.SPLIT, Col.LN_TREND, "sin_1"}
        assert expected <= set(panel.columns)
        assert set(panel[Col.SPLIT]) == {Split.TRAIN, Split.TEST}


class TestDataPrepFactory:
    """DataPrepFactory wires a ready-to-run service."""

    def test_builds_service(
        self, cfg: Config, loader_settings: LoaderSettings, tmp_path: Path
    ) -> None:
        """Returns a DataPrepService."""
        service = DataPrepFactory(cfg, loader_settings, tmp_path / "p.parquet").build()
        assert isinstance(service, DataPrepService)
