"""Optimization layer (Phase 4): profit-maximising regular prices with NLopt.

inputs.py (P0, base volume, elasticity, cost, pack size), demand.py (constant and
linear curves), problem.py (objectives, constraints, KPIs), solver.py (NLopt,
rounding), robustness.py (posterior draws), service.py (facade + factory).
"""

from src.optimization.service import (
    OptimizationFactory,
    OptimizationOutputs,
    OptimizationReport,
    OptimizationService,
)

__all__ = [
    "OptimizationFactory",
    "OptimizationOutputs",
    "OptimizationReport",
    "OptimizationService",
]
