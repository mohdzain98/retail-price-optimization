"""Tests for src.optimization.service: the facade runs end to end and writes outputs.

Covers both objectives side by side and the rounding safeguard (rounded prices that
score below current are replaced by current prices). One test class per production class.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config, Objective
from src.optimization import OptimizationFactory, OptimizationOutputs, OptimizationService
from src.schema import Col, Metric, Scenario, kpi_col

AT_OPTIMUM_BETA = -1 / 0.35  # with a 35% margin, P* = c*beta/(1+beta) equals the current price


def _config(**optimization: object) -> Config:
    """Shipped config with a small robustness sample and optional overrides."""
    cfg = Config.load()
    opt = cfg.optimization.model_copy(update={"robustness_draws": 5, **optimization})
    return cfg.model_copy(update={"optimization": opt})


def _outputs(tmp_path: Path) -> OptimizationOutputs:
    """Output paths under tmp_path."""
    return OptimizationOutputs(tmp_path / "rec.parquet", tmp_path / "summary.parquet")


class TestOptimizationService:
    """OptimizationService optimises every objective, scores and writes."""

    def test_both_objectives(
        self, baseline_and_final: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
    ) -> None:
        """One block per objective; each never loses on its own objective; no violations."""
        baseline, final = baseline_and_final
        outputs = _outputs(tmp_path)
        report = (
            OptimizationFactory(_config(), outputs).build().run(baseline, final, pd.DataFrame())
        )
        assert outputs.recommendations.exists() and outputs.summary.exists()
        recs = report.recommendations
        assert set(recs[Col.OBJECTIVE]) == {Objective.PROFIT.value, Objective.REVENUE.value}
        assert len(recs) == 2 * 2
        rows_per_objective = len(_config().optimization.algorithms) + 2
        assert len(report.summary) == 2 * rows_per_objective
        rec = report.summary[report.summary[Col.SCENARIO] == Scenario.RECOMMENDED.value]
        by_objective = rec.set_index(Col.OBJECTIVE)
        assert by_objective.loc[Objective.PROFIT.value, "profit_change"] >= -1e-9
        assert by_objective.loc[Objective.REVENUE.value, "revenue_change"] >= -1e-9
        assert (report.summary[Col.MAX_VIOLATION] <= 1e-6).all()
        assert {Col.RULE_STATUS, Col.EXPECTED_DIRECTION, Col.OWN_OPTIMAL_PRICE} <= set(recs.columns)

    def test_rounding_safeguard(
        self, baseline_and_final: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
    ) -> None:
        """Optimum = current price, only .49 endings, loose cap/floor (so repair keeps the bad
        rounding feasible): the rounded prices lose profit, so current prices are kept."""
        baseline, final = baseline_and_final
        final = final.assign(**{Col.ELASTICITY: AT_OPTIMUM_BETA})
        cfg = _config(
            objectives=[Objective.PROFIT],
            price_endings=[0.49],
            max_avg_price_increase=0.5,
            volume_floor=0.5,
        )
        report = (
            OptimizationFactory(cfg, _outputs(tmp_path))
            .build()
            .run(baseline, final, pd.DataFrame())
        )
        recs = report.recommendations
        assert recs[Col.SAFEGUARD_APPLIED].all()
        pd.testing.assert_series_equal(
            recs[kpi_col(Scenario.RECOMMENDED, Metric.PRICE)],
            recs[Col.CURRENT_PRICE],
            check_names=False,
        )


class TestOptimizationFactory:
    """OptimizationFactory wires a ready-to-run service."""

    def test_builds_service(self, tmp_path: Path) -> None:
        """Returns an OptimizationService."""
        assert isinstance(
            OptimizationFactory(_config(), _outputs(tmp_path)).build(), OptimizationService
        )
