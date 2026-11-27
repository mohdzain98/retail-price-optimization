"""Tests for src.optimization.service: the facade runs end to end and writes outputs.

Uses the small baseline/final fixture (two current products, one discontinued) and
the shipped config with fewer robustness draws. One test class per production class.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config
from src.optimization import OptimizationFactory, OptimizationOutputs, OptimizationService
from src.schema import Col, Scenario


def _config() -> Config:
    """Shipped config with a small robustness sample."""
    cfg = Config.load()
    return cfg.model_copy(
        update={"optimization": cfg.optimization.model_copy(update={"robustness_draws": 5})}
    )


class TestOptimizationService:
    """OptimizationService optimises, scores and writes."""

    def test_run(
        self, baseline_and_final: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
    ) -> None:
        """Both tables written; summary has one row per algorithm + 2 scenarios; no violations."""
        baseline, final = baseline_and_final
        outputs = OptimizationOutputs(tmp_path / "rec.parquet", tmp_path / "summary.parquet")
        report = (
            OptimizationFactory(_config(), outputs).build().run(baseline, final, pd.DataFrame())
        )
        assert outputs.recommendations.exists() and outputs.summary.exists()
        assert len(report.recommendations) == 2
        n_algorithms = len(_config().optimization.algorithms)
        assert len(report.summary) == n_algorithms + 2
        assert set(report.summary[Col.SCENARIO]) >= {
            Scenario.RECOMMENDED.value,
            Scenario.LINEAR_FORMULA.value,
        }
        assert (report.summary["profit_change"] >= -1e-9).all()
        assert (report.summary[Col.MAX_VIOLATION] <= 1e-6).all()


class TestOptimizationFactory:
    """OptimizationFactory wires a ready-to-run service."""

    def test_builds_service(self, tmp_path: Path) -> None:
        """Returns an OptimizationService."""
        outputs = OptimizationOutputs(tmp_path / "a.parquet", tmp_path / "b.parquet")
        assert isinstance(OptimizationFactory(_config(), outputs).build(), OptimizationService)
