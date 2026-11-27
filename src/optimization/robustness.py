"""Robustness of the optimal prices to elasticity uncertainty.

Re-solves the problem for a sample of posterior elasticity draws (Bayesian model)
and reports the 5th/50th/95th percentile of each product's optimal price. Products
whose final elasticity did not come from the Bayesian model keep their fixed value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ModelName, Objective
from src.optimization.demand import DemandCurve
from src.optimization.inputs import OptimizationInputs
from src.optimization.problem import ProblemBuilder
from src.optimization.solver import NloptSolver
from src.schema import Col, ElasticitySource
from src.utils import AppLogger

logger = AppLogger.get_logger(__name__)

DRAW = "draw"
PERCENTILES: dict[Col, float] = {Col.ROBUST_P05: 5, Col.ROBUST_P50: 50, Col.ROBUST_P95: 95}


class RobustnessAnalyzer:
    """Optimal-price percentiles across posterior elasticity draws.

    Each draw is a full re-optimization with the portfolio constraints active.
    """

    def __init__(
        self, builder: ProblemBuilder, solver: NloptSolver, n_draws: int, seed: int
    ) -> None:
        self._builder = builder
        self._solver = solver
        self._n_draws = n_draws
        self._seed = seed

    def run(
        self,
        inputs: OptimizationInputs,
        demand: DemandCurve,
        draws: pd.DataFrame,
        objective: Objective | None = None,
    ) -> pd.DataFrame:
        """UPC + p05/p50/p95 optimal prices for the objective; empty if no draws exist."""
        if draws.empty:
            return pd.DataFrame(columns=[Col.UPC, *PERCENTILES])
        matrix = self._draw_matrix(inputs, draws)
        prices = []
        for beta in matrix:
            problem = self._builder.build(inputs.with_beta(beta), demand, objective)
            result = self._solver.solve(problem)
            prices.append(result.ratios * inputs.p0)
        stacked = np.vstack(prices)
        out = {col: np.percentile(stacked, q, axis=0) for col, q in PERCENTILES.items()}
        logger.info("Robustness done", extra={"draws": len(matrix)})
        return pd.DataFrame({Col.UPC: inputs.products[Col.UPC].to_numpy(), **out})

    def _draw_matrix(self, inputs: OptimizationInputs, draws: pd.DataFrame) -> np.ndarray:
        """n_draws x n_products elasticities; non-Bayes products keep their final value."""
        wide = draws.pivot_table(index=DRAW, columns=Col.UPC, values=Col.ELASTICITY)
        rng = np.random.default_rng(self._seed)
        chosen = rng.choice(
            wide.index.to_numpy(), size=min(self._n_draws, len(wide)), replace=False
        )
        products = inputs.products
        sampled = wide.loc[chosen].reindex(columns=products[Col.UPC]).to_numpy()
        uses_bayes = (
            (products[Col.MODEL] == ModelName.BAYES_HIERARCHICAL.value)
            & (products[Col.ELASTICITY_SOURCE] == ElasticitySource.ESTIMATED.value)
        ).to_numpy()
        fixed = np.broadcast_to(inputs.beta, sampled.shape)
        return np.where(uses_bayes & ~np.isnan(sampled), sampled, fixed)
