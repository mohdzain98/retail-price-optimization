"""Tests for src.promo.service: the facade runs end to end and writes every table.

Uses the synthetic panel, a Phase 4-style recommendations table and the stub lift
model; checks scenarios, rule compliance and that the MILP never trails greedy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config, ModelName, Objective
from src.promo import PromoFactory, PromoOutputs
from src.promo.planner import PlanRules
from src.schema import Col, PromoScenario
from tests.promo.conftest import LinearLift, make_recommendations


def _config() -> Config:
    """Shipped config with a short horizon and few frontier points (fast)."""
    cfg = Config.load()
    promo = cfg.promo.model_copy(update={"frontier_points": 4, "budget": 20_000.0})
    forecast = cfg.forecast.model_copy(update={"horizon_weeks": 8})
    return cfg.model_copy(update={"promo": promo, "forecast": forecast})


class TestPromoService:
    """PromoService plans every objective and writes its outputs."""

    def test_end_to_end(self, panel: pd.DataFrame, tmp_path: Path) -> None:
        """All tables written; six scenarios per objective; plans obey the rules."""
        cfg = _config()
        outputs = PromoOutputs(*(tmp_path / f"{n}.parquet" for n in range(6)))
        final = pd.DataFrame({Col.UPC: panel[Col.UPC].unique(), Col.PROMO_ELASTICITY: -2.5})
        baseline = panel.assign(**{Col.BASELINE: panel[Col.UNITS] * 0.8})
        models = {ModelName.BAYES_HIERARCHICAL: LinearLift()}
        report = (
            PromoFactory(cfg, models, outputs)
            .build()
            .run(panel, baseline, final, make_recommendations(panel))
        )
        assert all(Path(p).exists() for p in vars(outputs).values())
        assert len(report.summary) == 2 * len(PromoScenario)
        assert set(report.summary[Col.OBJECTIVE]) == {Objective.PROFIT, Objective.REVENUE}
        rules = PlanRules.from_config(cfg.promo)
        for (_, scenario), plan in report.plan.groupby([Col.OBJECTIVE, Col.SCENARIO]):
            if scenario != PromoScenario.LAST_YEAR_PLAN.value:
                assert rules.violations(plan, plan[Col.BUDGET].iloc[0]) == []
        gain = (
            report.plan.groupby([Col.OBJECTIVE, Col.SCENARIO])[Col.GAIN]
            .sum()
            .unstack(fill_value=0.0)
        )
        milp, greedy = PromoScenario.MILP_PLAN.value, PromoScenario.GREEDY_PLAN.value
        assert (gain[milp] >= gain[greedy] - 1e-6).all()
        assert report.frontier.groupby(Col.OBJECTIVE)[Col.RECOMMENDED_BUDGET].sum().eq(1).all()
        assert report.forecast[Col.HORIZON_WEEK].max() == 8
