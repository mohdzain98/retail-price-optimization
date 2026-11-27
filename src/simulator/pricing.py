"""Live regular-price simulation: re-optimize with user guardrails, or score any prices.

PriceSimulator rebuilds the Phase 4 optimizer inputs from the stored recommendations,
lets the user change margins and guardrails, re-solves with NLopt (rounded, repaired,
safeguarded), and scores user-edited prices against the same demand curve.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config, Objective, OptimizationConfig
from src.optimization.demand import DEMAND_CURVES, DemandCurve
from src.optimization.inputs import ELASTICITY_ATTRS, PRODUCT_ATTRS, OptimizationInputs
from src.optimization.problem import (
    CONSTRAINTS,
    OBJECTIVES,
    KpiCalculator,
    PriceLadder,
    PricingProblem,
    ProblemBuilder,
)
from src.optimization.solver import (
    FeasibilityRepair,
    NloptSolver,
    PricePointRounder,
    PriceRecommender,
)
from src.schema import Col, Metric, Scenario, kpi_col

INPUT_COLS: tuple[Col, ...] = (
    *PRODUCT_ATTRS,
    *ELASTICITY_ATTRS,
    Col.CURRENT_PRICE,
    Col.BASE_VOLUME,
    Col.SIZE_VALUE,
    Col.SIZE_UNIT,
)
METRICS: tuple[Metric, ...] = (Metric.VOLUME, Metric.REVENUE, Metric.PROFIT)
NEW = Scenario.RECOMMENDED  # KPI columns of the simulated prices
CURVE_RANGE, CURVE_POINTS = (0.7, 1.3), 121
VIOLATION_TOL = 1e-6


@dataclass(frozen=True)
class PriceSettings:
    """User guardrails and category margins (hashable, so results can be cached).

    `margins` is a tuple of (category, margin) pairs; unit cost = (1 - margin) x P0.
    """

    bounds: tuple[float, float]
    volume_floor: float
    max_avg_increase: float
    ladder: bool
    margins: tuple[tuple[str, float], ...]

    @classmethod
    def from_config(cls, cfg: Config, categories: list[str]) -> PriceSettings:
        """The pipeline's own guardrails and margins."""
        opt = cfg.optimization
        margins = tuple((c, cfg.cost.margin_for(c)) for c in categories)
        return cls(opt.price_bounds, opt.volume_floor, opt.max_avg_price_increase, True, margins)


@dataclass(frozen=True)
class PriceResult:
    """Per-product table, portfolio totals and diagnostics of one price scenario.

    `broken` lists guardrails the prices violate (empty for optimizer output).
    """

    products: pd.DataFrame
    totals: pd.DataFrame
    safeguard_applied: bool
    broken: tuple[str, ...]


class PriceSimulator:
    """Re-optimizes or scores regular prices with the Phase 4 machinery.

    Inputs (P0, base volume, elasticity, pack size) come from the stored
    recommendations; only margins and guardrails change.
    """

    def __init__(self, cfg: Config, recommendations: pd.DataFrame) -> None:
        self._cfg = cfg
        first = recommendations[
            recommendations[Col.OBJECTIVE] == recommendations[Col.OBJECTIVE].iloc[0]
        ]
        self._products = first[list(INPUT_COLS)].reset_index(drop=True)
        self._kpis = KpiCalculator()

    @property
    def categories(self) -> list[str]:
        """Product categories in sorted order."""
        return sorted(self._products[Col.CATEGORY].unique())

    @property
    def products(self) -> pd.DataFrame:
        """One row per product with the optimizer inputs (no unit cost)."""
        return self._products

    def inputs(self, margins: Mapping[str, float]) -> OptimizationInputs:
        """Optimizer inputs with unit cost = (1 - category margin) x current price."""
        margin = (
            self._products[Col.CATEGORY].map(dict(margins)).fillna(self._cfg.cost.default_margin)
        )
        cost = (1 - margin) * self._products[Col.CURRENT_PRICE]
        return OptimizationInputs(self._products.assign(**{Col.UNIT_COST: cost}))

    def optimize(self, objective: Objective, settings: PriceSettings) -> PriceResult:
        """Solve with the primary NLopt algorithm, round, repair and safeguard."""
        opt = self._opt_config(settings)
        problem = self._problem(opt, objective, settings)
        algorithm = opt.algorithms[0]
        solver = NloptSolver(algorithm, opt.maxeval, opt.xtol_rel, opt.max_seconds)
        rounder = PricePointRounder(opt.price_endings)
        recommender = PriceRecommender(rounder, FeasibilityRepair(rounder))
        recommendation = recommender.recommend(problem, solver.solve(problem).ratios)
        return self._result(problem, recommendation.ratios, recommendation.safeguard_applied)

    def score(
        self, objective: Objective, settings: PriceSettings, prices: np.ndarray
    ) -> PriceResult:
        """KPIs of user-chosen prices (aligned to `products`) and the guardrails they break."""
        problem = self._problem(self._opt_config(settings), objective, settings)
        return self._result(problem, np.asarray(prices, float) / problem.inputs.p0, False)

    def curve(self, objective: Objective, settings: PriceSettings, upc: int) -> pd.DataFrame:
        """Weekly objective value (P - c_eff) Q over 70-130% of one product's current price."""
        opt = self._opt_config(settings)
        inputs = self.inputs(dict(settings.margins))
        i = int(np.flatnonzero(self._products[Col.UPC].to_numpy() == upc)[0])
        ratios = np.linspace(*CURVE_RANGE, CURVE_POINTS)
        c_eff = OBJECTIVES.get(objective).from_config(opt).effective_cost(inputs)[i]
        volume = self._demand(opt).volume(
            ratios,
            np.full_like(ratios, inputs.beta[i]),
            np.full_like(ratios, inputs.base_volume[i]),
        )
        price = ratios * inputs.p0[i]
        return pd.DataFrame({Col.PRICE: price, Col.OBJECTIVE_VALUE: (price - c_eff) * volume})

    def _result(self, problem: PricingProblem, ratios: np.ndarray, safeguard: bool) -> PriceResult:
        """Product table, totals and broken guardrails for relative prices `ratios`."""
        inputs, demand = problem.inputs, problem.demand
        current = self._kpis.evaluate(inputs, demand, np.ones(problem.n))
        new = self._kpis.evaluate(inputs, demand, ratios)
        table = inputs.products[
            [
                Col.UPC,
                Col.DESCRIPTION,
                Col.PRODUCT_SIZE,
                Col.CATEGORY,
                Col.ELASTICITY,
                Col.UNIT_COST,
                Col.CURRENT_PRICE,
            ]
        ].assign(
            **{kpi_col(NEW, Metric.PRICE): new[Metric.PRICE], Col.PRICE_RATIO: ratios - 1},
            **{kpi_col(Scenario.CURRENT, m): current[m] for m in METRICS},
            **{kpi_col(NEW, m): new[m] for m in METRICS},
        )
        totals = pd.DataFrame(
            {
                Scenario.CURRENT.value: [current[m].sum() for m in METRICS],
                NEW.value: [new[m].sum() for m in METRICS],
            },
            index=[m.value for m in METRICS],
        )
        totals["change"] = totals[NEW.value] / totals[Scenario.CURRENT.value] - 1
        return PriceResult(table, totals, safeguard, self._broken(problem, ratios))

    @staticmethod
    def _broken(problem: PricingProblem, ratios: np.ndarray) -> tuple[str, ...]:
        """Names of guardrails violated at `ratios` (price bounds included)."""
        low, high = problem.bounds
        broken = ["price_bounds"] if ((ratios < low - 1e-9) | (ratios > high + 1e-9)).any() else []
        for constraint in problem.constraints:
            if constraint.size:
                values, _ = constraint.evaluate(ratios, problem.inputs, problem.demand)
                if (values > VIOLATION_TOL).any():
                    broken.append(constraint.name)
        return tuple(broken)

    def _problem(
        self, opt: OptimizationConfig, objective: Objective, settings: PriceSettings
    ) -> PricingProblem:
        """Pricing problem with the user's margins and guardrails (ladder optional)."""
        names = [c.name for c in CONSTRAINTS if settings.ladder or c is not PriceLadder]
        builder = ProblemBuilder(opt, names)
        return builder.build(self.inputs(dict(settings.margins)), self._demand(opt), objective)

    def _opt_config(self, settings: PriceSettings) -> OptimizationConfig:
        """Pipeline optimization config with the user's guardrails."""
        return self._cfg.optimization.model_copy(
            update={
                "price_bounds": settings.bounds,
                "volume_floor": settings.volume_floor,
                "max_avg_price_increase": settings.max_avg_increase,
            }
        )

    @staticmethod
    def _demand(opt: OptimizationConfig) -> DemandCurve:
        """The configured demand curve."""
        return DEMAND_CURVES.get(opt.demand_form)()
