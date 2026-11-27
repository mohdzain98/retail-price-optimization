"""Tests for src.simulator.data: loading the processed outputs the app reads.

Uses the session `processed_dir` fixture (real Phase 4/5 services on synthetic data).
One test class per production class.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.simulator.data import DATA_DIR_ENV, SimulatorData
from src.utils import MissingFileError


class TestSimulatorData:
    """SimulatorData loads every table or names the missing make targets."""

    def test_load(self, processed_dir: Path) -> None:
        """All tables load; both objectives are listed in stored order."""
        data = SimulatorData.load(processed_dir)
        assert data.objectives == ["profit", "revenue"]
        assert not data.candidates.empty and not data.forecast.empty

    def test_missing_promo_outputs(self, processed_dir: Path, tmp_path: Path) -> None:
        """Without the promo files the error asks for `make promo` only."""
        partial = tmp_path / "partial"
        shutil.copytree(processed_dir, partial)
        (partial / "promo_plan.parquet").unlink()
        with pytest.raises(MissingFileError, match=r"`make promo`") as err:
            SimulatorData.load(partial)
        assert "optimize" not in str(err.value)

    def test_directory_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """RPO_DATA_DIR overrides the default directory."""
        monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
        assert SimulatorData.directory() == tmp_path
