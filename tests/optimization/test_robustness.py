"""Tests for src.optimization.robustness.RobustnessAnalyzer.

Products estimated by the Bayesian model get a price range from posterior draws;
products whose elasticity came from another model keep a single optimal price.
One test class per production class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Algorithm, ModelName, OptimizationConfig
from src.optimization.demand import ConstantElasticity
from src.optimization.problem import ProblemBuilder
from src.optimization.robustness import RobustnessAnalyzer
from src.optimization.solver import NloptSolver
from src.schema import Col
from tests.optimization.conftest import make_inputs


class TestRobustnessAnalyzer:
    """Percentiles of optimal prices across draws."""

    def test_ranges(self, opt_cfg: OptimizationConfig) -> None:
        """Bayes product: p05 < p95; OLS-selected product: p05 == p95."""
        inputs = make_inputs([-2.5, -2.0], p0=[3.0, 3.0], cost_share=0.5)
        inputs = type(inputs)(inputs.products.assign(**{Col.MANUFACTURER: "BRAND"}))
        products = inputs.products.assign(
            **{Col.MODEL: [ModelName.BAYES_HIERARCHICAL.value, ModelName.OLS_PER_PRODUCT.value]}
        )
        inputs = type(inputs)(products)
        rng = np.random.default_rng(0)
        draws = pd.DataFrame(
            {
                "draw": np.repeat(np.arange(50), 2),
                Col.UPC: np.tile([1, 2], 50),
                Col.ELASTICITY: rng.normal(-2.5, 0.3, 100),
            }  # fmt: skip
        )
        cfg = opt_cfg.model_copy(update={"price_bounds": (0.5, 2.0)})
        solver = NloptSolver(Algorithm.LD_SLSQP, 2000, 1e-8)
        analyzer = RobustnessAnalyzer(ProblemBuilder(cfg, constraint_names=[]), solver, 30, 0)
        out = analyzer.run(inputs, ConstantElasticity(), draws).set_index(Col.UPC)
        assert out.loc[1, Col.ROBUST_P05] < out.loc[1, Col.ROBUST_P95]
        assert out.loc[2, Col.ROBUST_P05] == out.loc[2, Col.ROBUST_P95]

    def test_no_draws(self, opt_cfg: OptimizationConfig) -> None:
        """No draws gives an empty frame with the expected columns."""
        analyzer = RobustnessAnalyzer(
            ProblemBuilder(opt_cfg), NloptSolver(Algorithm.LD_SLSQP, 10, 1e-6), 5, 0
        )
        out = analyzer.run(make_inputs([-2.0]), ConstantElasticity(), pd.DataFrame())
        assert out.empty and Col.ROBUST_P50 in out.columns
