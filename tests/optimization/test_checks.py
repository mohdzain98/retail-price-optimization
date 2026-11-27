"""Tests for src.optimization.checks.DirectionRuleCheck.

Revenue must reproduce the RGMx rule (elastic -> cut, inelastic -> raise); profit
uses the margin-aware rule (raise iff |beta| < 1 / margin). Overrides are flagged.
One test class per production class.
"""

from __future__ import annotations

import numpy as np

from src.config import Objective, OptimizationConfig
from src.optimization.checks import DirectionRuleCheck
from src.optimization.demand import ConstantElasticity
from src.optimization.problem import ProblemBuilder
from src.schema import Col, Direction, RuleStatus
from tests.optimization.conftest import make_inputs

BETAS = [-0.6, -1.5, -4.0]  # inelastic, mildly elastic, very elastic; margin 40% -> 1/m = 2.5


def _check(cfg: OptimizationConfig, objective: Objective, ratios: np.ndarray) -> dict[str, list]:
    """Rule-check columns for three branded products at 40% margin."""
    inputs = make_inputs(BETAS, p0=[3.0, 3.0, 3.0], cost_share=0.6)
    inputs = type(inputs)(inputs.products.assign(**{Col.MANUFACTURER: "BRAND"}))
    problem = ProblemBuilder(cfg, constraint_names=[]).build(
        inputs, ConstantElasticity(), objective
    )
    out = DirectionRuleCheck().check(problem, ratios)
    return {c: out[c].tolist() for c in (Col.EXPECTED_DIRECTION, Col.RULE_STATUS)}


class TestDirectionRuleCheck:
    """Expected direction follows the objective's own-optimum rule."""

    def test_revenue_is_the_rgmx_rule(self, opt_cfg: OptimizationConfig) -> None:
        """Revenue: inelastic -> raise, elastic -> cut (regardless of margin)."""
        out = _check(opt_cfg, Objective.REVENUE, np.array([1.2, 0.8, 0.8]))
        assert out[Col.EXPECTED_DIRECTION] == [Direction.RAISE, Direction.CUT, Direction.CUT]
        assert set(out[Col.RULE_STATUS]) == {RuleStatus.FOLLOWS}

    def test_profit_is_margin_aware(self, opt_cfg: OptimizationConfig) -> None:
        """Profit at 40% margin: beta -1.5 (|beta| < 2.5) -> raise; beta -4 -> cut."""
        out = _check(opt_cfg, Objective.PROFIT, np.array([1.2, 1.2, 1.0]))
        assert out[Col.EXPECTED_DIRECTION] == [Direction.RAISE, Direction.RAISE, Direction.CUT]
        assert out[Col.RULE_STATUS] == [
            RuleStatus.FOLLOWS, RuleStatus.FOLLOWS, RuleStatus.CONSTRAINT_OVERRIDE,
        ]  # fmt: skip
