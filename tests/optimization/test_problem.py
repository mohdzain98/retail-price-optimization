"""Tests for src.optimization.problem: objectives, constraints, builder and KPIs.

Every analytic gradient / Jacobian is checked against finite differences, and the
price ladder must only enforce pairs that hold at current prices.
One test class per production class.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from src.config import Objective, OptimizationConfig
from src.optimization.demand import ConstantElasticity
from src.optimization.problem import (
    KpiCalculator,
    MaxAveragePriceIncrease,
    PriceLadder,
    ProblemBuilder,
    ProfitObjective,
    RevenueObjective,
    VolumeFloor,
    WeightedObjective,
)
from src.schema import Metric
from tests.optimization.conftest import make_inputs

R = np.array([0.9, 1.05, 1.15])
DEMAND = ConstantElasticity()


def _fd_grad(fn: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central finite-difference gradient of a scalar function."""
    eye = np.eye(len(x)) * h
    return np.array([(fn(x + e) - fn(x - e)) / (2 * h) for e in eye])


@pytest.mark.parametrize(
    "objective", [ProfitObjective(), RevenueObjective(), WeightedObjective(0.5)]
)
def test_objective_gradients(objective: ProfitObjective) -> None:
    """Analytic objective gradients match finite differences."""
    inputs = make_inputs([-2.0, -1.5, -0.8])
    _, grad = objective.value_and_grad(R, inputs, DEMAND)
    numeric = _fd_grad(lambda x: objective.value_and_grad(x, inputs, DEMAND)[0], R)
    np.testing.assert_allclose(grad, numeric, rtol=1e-5)


class TestProfitObjective:
    """Profit = sum (P - c) Q."""

    def test_value_at_current(self) -> None:
        """At r=1 profit = sum (P0 - 0.6 P0) * 100."""
        inputs = make_inputs([-2.0, -1.5])
        value, _ = ProfitObjective().value_and_grad(np.ones(2), inputs, DEMAND)
        assert value == pytest.approx(0.4 * (2.0 + 3.0) * 100)


class TestRevenueObjective:
    """Revenue = sum P Q."""

    def test_value_at_current(self) -> None:
        """At r=1 revenue = sum P0 * 100."""
        value, _ = RevenueObjective().value_and_grad(np.ones(2), make_inputs([-2.0, -1.5]), DEMAND)
        assert value == pytest.approx((2.0 + 3.0) * 100)


class TestWeightedObjective:
    """w * profit + (1 - w) * revenue = sum((P - w c) Q)."""

    def test_effective_costs(self) -> None:
        """Profit charges c, revenue 0, weighted w * c."""
        inputs = make_inputs([-2.0, -1.5])
        np.testing.assert_allclose(ProfitObjective().effective_cost(inputs), inputs.cost)
        np.testing.assert_allclose(RevenueObjective().effective_cost(inputs), 0.0)
        np.testing.assert_allclose(WeightedObjective(0.5).effective_cost(inputs), 0.5 * inputs.cost)

    def test_blend(self) -> None:
        """w = 0.5 is the midpoint of profit and revenue."""
        inputs = make_inputs([-2.0, -1.5])
        value, _ = WeightedObjective(0.5).value_and_grad(np.ones(2), inputs, DEMAND)
        assert value == pytest.approx(0.5 * 200 + 0.5 * 500)


class TestVolumeFloor:
    """floor - sum Q(r) / sum Q(1) <= 0."""

    def test_value_and_jacobian(self) -> None:
        """Feasible at r=1; Jacobian matches finite differences."""
        inputs = make_inputs([-2.0, -1.5, -0.8])
        constraint = VolumeFloor(0.95)
        assert constraint.evaluate(np.ones(3), inputs, DEMAND)[0][0] == pytest.approx(-0.05)
        _, jac = constraint.evaluate(R, inputs, DEMAND)
        numeric = _fd_grad(lambda x: constraint.evaluate(x, inputs, DEMAND)[0][0], R)
        np.testing.assert_allclose(jac[0], numeric, rtol=1e-5)


class TestMaxAveragePriceIncrease:
    """sum w r - (1 + cap) <= 0 with revenue weights."""

    def test_weights_and_value(self, opt_cfg: OptimizationConfig) -> None:
        """Weights sum to 1; a uniform +5% increase is exactly on the boundary."""
        inputs = make_inputs([-2.0, -1.5, -0.8])
        constraint = MaxAveragePriceIncrease.from_config(opt_cfg, inputs)
        values, jac = constraint.evaluate(
            np.full(3, 1 + opt_cfg.max_avg_price_increase), inputs, DEMAND
        )
        assert jac.sum() == pytest.approx(1.0) and values[0] == pytest.approx(0.0, abs=1e-12)


class TestPriceLadder:
    """Branded unit price >= (1 + gap) x private-label unit price."""

    def test_pairs_and_value(self, opt_cfg: OptimizationConfig) -> None:
        """PL at $2 vs brands at $3 (gap 10%) form 2 pairs; violated if PL rises 50%."""
        inputs = make_inputs([-2.0, -1.5, -0.8])
        ladder = PriceLadder.from_config(opt_cfg, inputs)
        assert ladder.size == 2
        assert (ladder.evaluate(np.ones(3), inputs, DEMAND)[0] < 0).all()
        assert (ladder.evaluate(np.array([1.5, 1.0, 1.0]), inputs, DEMAND)[0] > 0).all()

    def test_skips_pairs_that_do_not_hold(self, opt_cfg: OptimizationConfig) -> None:
        """A PL priced above the brand today is not forced into a ladder."""
        inputs = make_inputs([-2.0, -1.5], p0=[3.5, 3.0])
        assert PriceLadder.from_config(opt_cfg, inputs).size == 0


class TestProblemBuilder:
    """ProblemBuilder assembles configured pieces."""

    def test_builds_all_constraints(self, opt_cfg: OptimizationConfig) -> None:
        """All registered constraints are present by default; none when told so."""
        inputs = make_inputs([-2.0, -1.5])
        assert len(ProblemBuilder(opt_cfg).build(inputs, DEMAND).constraints) == 3
        revenue = ProblemBuilder(opt_cfg).build(inputs, DEMAND, Objective.REVENUE)
        assert isinstance(revenue.objective, RevenueObjective)
        assert ProblemBuilder(opt_cfg, constraint_names=[]).build(inputs, DEMAND).constraints == ()


class TestKpiCalculator:
    """Weekly price, volume, revenue and profit per product."""

    def test_current_kpis(self) -> None:
        """At r=1 volume is the base volume and profit is 40% of revenue."""
        kpis = KpiCalculator().evaluate(make_inputs([-2.0, -1.5]), DEMAND, np.ones(2))
        np.testing.assert_allclose(kpis[Metric.VOLUME], [100.0, 100.0])
        np.testing.assert_allclose(kpis[Metric.PROFIT], 0.4 * kpis[Metric.REVENUE])
