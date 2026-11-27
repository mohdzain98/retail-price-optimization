"""Pricing-direction rule check: does each recommendation follow the objective's rule?

A product's own optimum (no portfolio constraints) is P* = c_eff * beta / (1 + beta)
for beta < -1, else the upper bound. With revenue (c_eff = 0) this is the classic revenue rule:
elastic -> cut, inelastic -> raise. With profit it is margin-aware (|beta| vs 1/margin).
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from src.optimization.problem import PricingProblem
from src.schema import Col, Direction, RuleStatus


class DirectionRuleCheck:
    """Compares each product's expected direction with the recommended one.

    A mismatch means a portfolio constraint (cap, ladder, volume floor) overrode the
    product's own optimum; it is reported, not treated as an error.
    """

    HOLD_TOL: ClassVar[float] = 0.005  # moves within +/-0.5% count as hold

    def check(self, problem: PricingProblem, ratios: np.ndarray) -> pd.DataFrame:
        """One row per product: own optimum, effective cost, expected/actual direction, status."""
        inputs = problem.inputs
        own = self.own_optimum_ratio(problem)
        expected, actual = self._direction(own), self._direction(ratios)
        return pd.DataFrame(
            {
                Col.UPC: inputs.products[Col.UPC].to_numpy(),
                Col.EFFECTIVE_COST: problem.objective.effective_cost(inputs),
                Col.OWN_OPTIMAL_PRICE: own * inputs.p0,
                Col.EXPECTED_DIRECTION: expected,
                Col.ACTUAL_DIRECTION: actual,
                Col.RULE_STATUS: np.where(
                    expected == actual,
                    RuleStatus.FOLLOWS.value,
                    RuleStatus.CONSTRAINT_OVERRIDE.value,
                ),
            }
        )

    @staticmethod
    def own_optimum_ratio(problem: PricingProblem) -> np.ndarray:
        """Single-product optimum r* = c_eff beta / ((1 + beta) P0), clipped to the bounds.

        For beta >= -1 the objective rises with price, so r* is the upper bound
        (beta = -1 with zero cost is flat: r* = 1).
        """
        inputs, (low, high) = problem.inputs, problem.bounds
        beta, c_eff = inputs.beta, problem.objective.effective_cost(inputs)
        elastic = beta < -1
        safe = np.where(elastic, 1 + beta, -1.0)  # avoid dividing by zero where unused
        interior = c_eff * beta / (safe * inputs.p0)
        flat = np.isclose(beta, -1) & np.isclose(c_eff, 0)
        own = np.where(elastic, interior, np.where(flat, 1.0, high))
        return np.clip(own, low, high)

    def _direction(self, ratios: np.ndarray) -> np.ndarray:
        """raise / cut / hold labels for relative prices."""
        return np.select(
            [ratios > 1 + self.HOLD_TOL, ratios < 1 - self.HOLD_TOL],
            [Direction.RAISE.value, Direction.CUT.value],
            default=Direction.HOLD.value,
        )
