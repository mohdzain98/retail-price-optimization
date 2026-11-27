"""The pricing problem: objective, constraints and KPIs over relative prices r = P / P0.

Objectives (profit, revenue, weighted) and constraints (volume floor, average
increase cap, private-label price ladder) register in their own registries and all
provide analytic gradients. Every constraint is written as fc(r) <= 0 (NLopt form).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from src.config import Objective, OptimizationConfig
from src.optimization.demand import DemandCurve
from src.optimization.inputs import OptimizationInputs
from src.schema import Col, Metric
from src.utils import Registry

PRIVATE_LABEL = "PRIVATE LABEL"


class ObjectiveFn(ABC):
    """Weekly portfolio value to maximise, with its gradient in r."""

    objective: ClassVar[Objective]

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: OptimizationConfig) -> ObjectiveFn:
        """Build from optimization config."""

    @abstractmethod
    def value_and_grad(
        self, r: np.ndarray, inputs: OptimizationInputs, demand: DemandCurve
    ) -> tuple[float, np.ndarray]:
        """(objective, d objective / d r)."""


OBJECTIVES: Registry[Objective, type[ObjectiveFn]] = Registry("objective")


def _revenue_profit(
    r: np.ndarray, inputs: OptimizationInputs, demand: DemandCurve
) -> tuple[float, np.ndarray, float, np.ndarray]:
    """Revenue, its gradient, profit and its gradient (shared by all objectives)."""
    p0, beta, base = inputs.p0, inputs.beta, inputs.base_volume
    price, q, dq = r * p0, demand.volume(r, beta, base), demand.slope(r, beta, base)
    margin = price - inputs.cost
    revenue, d_revenue = float(np.sum(price * q)), p0 * q + price * dq
    profit, d_profit = float(np.sum(margin * q)), p0 * q + margin * dq
    return revenue, d_revenue, profit, d_profit


@OBJECTIVES.register
class ProfitObjective(ObjectiveFn):
    """Maximise weekly gross profit sum((P - c) Q)."""

    objective = Objective.PROFIT

    @classmethod
    def from_config(cls, cfg: OptimizationConfig) -> ProfitObjective:
        """No settings."""
        return cls()

    def value_and_grad(
        self, r: np.ndarray, inputs: OptimizationInputs, demand: DemandCurve
    ) -> tuple[float, np.ndarray]:
        """Profit and gradient."""
        _, _, profit, d_profit = _revenue_profit(r, inputs, demand)
        return profit, d_profit


@OBJECTIVES.register
class RevenueObjective(ObjectiveFn):
    """Maximise weekly revenue sum(P Q); ignores cost (hits bounds when |beta| < 1)."""

    objective = Objective.REVENUE

    @classmethod
    def from_config(cls, cfg: OptimizationConfig) -> RevenueObjective:
        """No settings."""
        return cls()

    def value_and_grad(
        self, r: np.ndarray, inputs: OptimizationInputs, demand: DemandCurve
    ) -> tuple[float, np.ndarray]:
        """Revenue and gradient."""
        revenue, d_revenue, _, _ = _revenue_profit(r, inputs, demand)
        return revenue, d_revenue


@OBJECTIVES.register
class WeightedObjective(ObjectiveFn):
    """w * profit + (1 - w) * revenue: trades margin against top line."""

    objective = Objective.WEIGHTED

    def __init__(self, profit_weight: float) -> None:
        self._w = profit_weight

    @classmethod
    def from_config(cls, cfg: OptimizationConfig) -> WeightedObjective:
        """Read the profit weight."""
        return cls(cfg.profit_weight)

    def value_and_grad(
        self, r: np.ndarray, inputs: OptimizationInputs, demand: DemandCurve
    ) -> tuple[float, np.ndarray]:
        """Weighted value and gradient."""
        revenue, d_revenue, profit, d_profit = _revenue_profit(r, inputs, demand)
        return self._w * profit + (1 - self._w) * revenue, self._w * d_profit + (
            1 - self._w
        ) * d_revenue


class Constraint(ABC):
    """A block of inequality constraints fc(r) <= 0 with its Jacobian (m x n)."""

    name: ClassVar[str]

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: OptimizationConfig, inputs: OptimizationInputs) -> Constraint:
        """Build from config and the products (some constraints depend on them)."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of scalar constraints m (0 means inactive)."""

    @abstractmethod
    def evaluate(
        self, r: np.ndarray, inputs: OptimizationInputs, demand: DemandCurve
    ) -> tuple[np.ndarray, np.ndarray]:
        """(values of length m, Jacobian m x n)."""


CONSTRAINTS: Registry[str, type[Constraint]] = Registry("name")


@CONSTRAINTS.register
class VolumeFloor(Constraint):
    """Portfolio volume must stay >= floor x current: floor - sum Q(r) / sum Q(1) <= 0.

    Written relative to current volume so the constraint is O(1) for the solver.
    """

    name = "volume_floor"

    def __init__(self, floor: float) -> None:
        self._floor = floor

    @classmethod
    def from_config(cls, cfg: OptimizationConfig, inputs: OptimizationInputs) -> VolumeFloor:
        """Read the floor."""
        return cls(cfg.volume_floor)

    @property
    def size(self) -> int:
        """One scalar constraint."""
        return 1

    def evaluate(
        self, r: np.ndarray, inputs: OptimizationInputs, demand: DemandCurve
    ) -> tuple[np.ndarray, np.ndarray]:
        """Value and gradient of the volume shortfall."""
        beta, base = inputs.beta, inputs.base_volume
        current = demand.volume(np.ones_like(r), beta, base).sum()
        value = self._floor - demand.volume(r, beta, base).sum() / current
        return np.array([value]), -demand.slope(r, beta, base)[None, :] / current


@CONSTRAINTS.register
class MaxAveragePriceIncrease(Constraint):
    """Revenue-weighted average price change capped: sum w r - (1 + cap) <= 0."""

    name = "max_avg_price_increase"

    def __init__(self, cap: float, weights: np.ndarray) -> None:
        self._cap = cap
        self._weights = weights

    @classmethod
    def from_config(
        cls, cfg: OptimizationConfig, inputs: OptimizationInputs
    ) -> MaxAveragePriceIncrease:
        """Weights are current revenue shares."""
        revenue = inputs.p0 * inputs.base_volume
        return cls(cfg.max_avg_price_increase, revenue / revenue.sum())

    @property
    def size(self) -> int:
        """One scalar constraint."""
        return 1

    def evaluate(
        self, r: np.ndarray, inputs: OptimizationInputs, demand: DemandCurve
    ) -> tuple[np.ndarray, np.ndarray]:
        """Linear constraint value and constant gradient."""
        return np.array([self._weights @ r - (1 + self._cap)]), self._weights[None, :]


@CONSTRAINTS.register
class PriceLadder(Constraint):
    """Branded unit price >= (1 + gap) x private-label unit price, same sub-category and unit.

    Only ladders that hold at current prices are enforced. Each row is divided by the
    branded unit price so values are relative (O(1)) for the solver.
    """

    name = "price_ladder"

    def __init__(self, gap: float, pairs: np.ndarray, unit_p0: np.ndarray, n: int) -> None:
        self._gap = gap
        self._pairs = pairs  # rows: (private_label_index, branded_index)
        self._unit_p0 = unit_p0
        self._n = n

    @classmethod
    def from_config(cls, cfg: OptimizationConfig, inputs: OptimizationInputs) -> PriceLadder:
        """Find private-label / branded pairs that currently satisfy the ladder."""
        products = inputs.products
        unit_p0 = (products[Col.CURRENT_PRICE] / products[Col.SIZE_VALUE]).to_numpy(float)
        pairs = cls._find_pairs(products, unit_p0, cfg.ladder_min_gap)
        return cls(cfg.ladder_min_gap, pairs, unit_p0, len(products))

    @staticmethod
    def _find_pairs(products: pd.DataFrame, unit_p0: np.ndarray, gap: float) -> np.ndarray:
        """(private-label, branded) index pairs within a sub-category and size unit."""
        frame = products.assign(idx=np.arange(len(products)), unit_price=unit_p0).dropna(
            subset=[Col.SIZE_VALUE]
        )
        is_pl = frame[Col.MANUFACTURER] == PRIVATE_LABEL
        keys = [Col.SUB_CATEGORY, Col.SIZE_UNIT]
        merged = frame[is_pl].merge(frame[~is_pl], on=keys, suffixes=("_pl", "_br"))
        holds = merged["unit_price_br"] >= (1 + gap) * merged["unit_price_pl"]
        return merged.loc[holds, ["idx_pl", "idx_br"]].to_numpy(int).reshape(-1, 2)

    @property
    def size(self) -> int:
        """One constraint per enforced pair."""
        return len(self._pairs)

    def evaluate(
        self, r: np.ndarray, inputs: OptimizationInputs, demand: DemandCurve
    ) -> tuple[np.ndarray, np.ndarray]:
        """(1 + gap) * (unit_pl / unit_br) * r_pl - r_br <= 0 for each pair."""
        pl, br = self._pairs[:, 0], self._pairs[:, 1]
        ratio = (1 + self._gap) * self._unit_p0[pl] / self._unit_p0[br]
        values = ratio * r[pl] - r[br]
        jac = np.zeros((len(self._pairs), self._n))
        rows = np.arange(len(self._pairs))
        jac[rows, pl] = ratio
        jac[rows, br] = -1.0
        return values, jac


@dataclass(frozen=True)
class PricingProblem:
    """Everything a solver needs: inputs, demand curve, objective, constraints, bounds."""

    inputs: OptimizationInputs
    demand: DemandCurve
    objective: ObjectiveFn
    constraints: tuple[Constraint, ...]
    bounds: tuple[float, float]

    @property
    def n(self) -> int:
        """Number of products (decision variables)."""
        return len(self.inputs.products)

    def max_violation(self, r: np.ndarray) -> float:
        """Largest positive constraint value at r (0 when feasible)."""
        worst = 0.0
        for constraint in self.constraints:
            if constraint.size:
                values, _ = constraint.evaluate(r, self.inputs, self.demand)
                worst = max(worst, float(values.max()))
        return worst


class ProblemBuilder:
    """Assembles a PricingProblem from config and registries for given inputs/demand.

    Constraints are built per inputs because some (ladders, weights) depend on products.
    """

    def __init__(
        self, cfg: OptimizationConfig, constraint_names: Sequence[str] | None = None
    ) -> None:
        self._cfg = cfg
        self._names = list(constraint_names) if constraint_names is not None else CONSTRAINTS.keys()

    def build(self, inputs: OptimizationInputs, demand: DemandCurve) -> PricingProblem:
        """Problem with the configured objective and all registered (or named) constraints."""
        constraints = tuple(
            CONSTRAINTS.get(name).from_config(self._cfg, inputs) for name in self._names
        )
        return PricingProblem(
            inputs=inputs,
            demand=demand,
            objective=OBJECTIVES.get(self._cfg.objective).from_config(self._cfg),
            constraints=constraints,
            bounds=self._cfg.price_bounds,
        )


class KpiCalculator:
    """Weekly price, volume, revenue and profit per product at relative prices r."""

    def evaluate(
        self, inputs: OptimizationInputs, demand: DemandCurve, r: np.ndarray
    ) -> dict[Metric, np.ndarray]:
        """One array per metric, aligned to inputs.products."""
        price = r * inputs.p0
        volume = demand.volume(r, inputs.beta, inputs.base_volume)
        return {
            Metric.PRICE: price,
            Metric.VOLUME: volume,
            Metric.REVENUE: price * volume,
            Metric.PROFIT: (price - inputs.cost) * volume,
        }
