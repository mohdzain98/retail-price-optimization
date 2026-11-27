"""Tests for src.optimization.solver: closed-form optimum, constraint signs, rounding.

For constant elasticity the profit-maximising price is P* = c * beta / (1 + beta);
NLopt must recover it. A volume-floor case guards the constraint-sign bug of the
original notebook (fc(x) <= 0). One test class per production class.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import Algorithm, OptimizationConfig
from src.optimization.demand import ConstantElasticity
from src.optimization.problem import ProblemBuilder
from src.optimization.solver import FeasibilityRepair, NloptSolver, PricePointRounder
from tests.optimization.conftest import make_inputs

WIDE = {"price_bounds": (0.5, 2.0)}


class TestNloptSolver:
    """NloptSolver finds known optima and respects constraints."""

    @pytest.mark.parametrize("algorithm", [Algorithm.LD_SLSQP, Algorithm.LN_COBYLA])
    def test_closed_form_optimum(self, algorithm: Algorithm, opt_cfg: OptimizationConfig) -> None:
        """beta = -2.5, cost = 0.5 P0  ->  r* = 0.5 * 2.5 / 1.5 = 0.8333."""
        inputs = make_inputs([-2.5], p0=[3.0], cost_share=0.5)
        cfg = opt_cfg.model_copy(update=WIDE)
        problem = ProblemBuilder(cfg, constraint_names=[]).build(inputs, ConstantElasticity())
        result = NloptSolver(algorithm, 5000, 1e-10).solve(problem)
        assert result.ratios[0] == pytest.approx(0.5 * 2.5 / 1.5, abs=1e-3)

    def test_volume_floor_sign(self, opt_cfg: OptimizationConfig) -> None:
        """Inelastic product wants a higher price; a 100% volume floor must stop it at P0."""
        inputs = make_inputs([-0.5], p0=[3.0])
        cfg = opt_cfg.model_copy(update={**WIDE, "volume_floor": 1.0})
        problem = ProblemBuilder(cfg, constraint_names=["volume_floor"]).build(
            inputs, ConstantElasticity()
        )
        result = NloptSolver(Algorithm.LD_SLSQP, 5000, 1e-10).solve(problem)
        assert result.ratios[0] == pytest.approx(1.0, abs=1e-4)
        assert result.max_violation <= 1e-6

    def test_inelastic_hits_upper_bound(self, opt_cfg: OptimizationConfig) -> None:
        """Without constraints an inelastic product goes to the upper bound."""
        problem = ProblemBuilder(opt_cfg, constraint_names=[]).build(
            make_inputs([-0.5], p0=[3.0]), ConstantElasticity()
        )
        result = NloptSolver(Algorithm.LD_SLSQP, 5000, 1e-10).solve(problem)
        assert result.ratios[0] == pytest.approx(opt_cfg.price_bounds[1], abs=1e-6)


class TestPricePointRounder:
    """PricePointRounder snaps to endings within bounds."""

    def test_nearest_ending(self) -> None:
        """3.12 -> 3.09; 3.95 -> 3.99; bounds force 2.49 when 2.39 is below the floor."""
        rounder = PricePointRounder([0.09, 0.49, 0.99])
        lo, hi = np.array([0.0, 0.0, 2.45]), np.array([10.0, 10.0, 10.0])
        out = rounder.round(np.array([3.12, 3.95, 2.40]), lo, hi)
        np.testing.assert_allclose(out, [3.09, 3.99, 2.49])

    def test_step_toward(self) -> None:
        """Steps one price point toward the target, or lands on it."""
        rounder = PricePointRounder([0.09, 0.49, 0.99])
        assert rounder.step_toward(3.99, 3.20) == pytest.approx(3.49)
        assert rounder.step_toward(3.49, 3.40) == pytest.approx(3.40)


class TestFeasibilityRepair:
    """FeasibilityRepair undoes rounding that breaks a constraint."""

    def test_restores_average_cap(self, opt_cfg: OptimizationConfig) -> None:
        """Prices rounded 10% up break the +5% cap; repair steps back until it holds."""
        inputs = make_inputs([-1.0, -1.0], p0=[3.0, 3.0])
        inputs = type(inputs)(inputs.products.assign(manufacturer="BRAND"))
        problem = ProblemBuilder(opt_cfg, constraint_names=["max_avg_price_increase"]).build(
            inputs, ConstantElasticity()
        )
        rounder = PricePointRounder([0.09, 0.19, 0.29, 0.39, 0.49, 0.59, 0.69, 0.79, 0.89, 0.99])
        repaired = FeasibilityRepair(rounder).repair(problem, np.array([3.29, 3.29]))
        assert problem.max_violation(repaired / inputs.p0) <= 1e-7
        assert (repaired <= 3.29).all() and (repaired >= 3.0).all()
