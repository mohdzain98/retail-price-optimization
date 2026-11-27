"""Budget frontier: best achievable gain at each trade budget, and the budget to use.

Solves the planner from zero to the spend of the unconstrained plan. The marginal
return of each step is extra gain per extra budget dollar; the recommended budget
is the last step (within the available budget) that still clears the hurdle.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.promo.planner import PromoPlanner
from src.schema import Col
from src.utils import AppLogger

logger = AppLogger.get_logger(__name__)


class BudgetFrontier:
    """Traces gain against budget and picks the recommended budget.

    Diminishing returns come from the planner taking the best options first; the
    hurdle (minimum marginal return) stops spending once a dollar buys too little.
    """

    def __init__(self, planner: PromoPlanner, points: int, min_marginal_return: float) -> None:
        self._planner = planner
        self._points = points
        self._hurdle = min_marginal_return

    def trace(self, candidates: pd.DataFrame, available: float) -> pd.DataFrame:
        """One row per budget: spend, gain, promos, marginal return, recommended flag."""
        full = self._planner.plan(candidates, math.inf)
        ceiling = float(candidates.loc[full, Col.TRADE_SPEND].sum())
        grid = np.linspace(0.0, ceiling, self._points)
        budgets = np.unique(np.append(grid, available) if available < ceiling else grid)
        rows = []
        for budget in budgets:
            chosen = candidates[self._planner.plan(candidates, float(budget))]
            rows.append({Col.BUDGET: budget, Col.TRADE_SPEND: chosen[Col.TRADE_SPEND].sum(),
                         Col.GAIN: chosen[Col.GAIN].sum(), Col.N_PROMOS: len(chosen)})  # fmt: skip
        frame = pd.DataFrame(rows)
        steps = frame[Col.BUDGET].diff()
        frame[Col.MARGINAL_RETURN] = frame[Col.GAIN].diff() / steps.where(steps > 0)
        frame[Col.RECOMMENDED_BUDGET] = frame.index == self._recommended(frame, available)
        return frame

    def _recommended(self, frame: pd.DataFrame, available: float) -> int:
        """Row of the last budget within `available` before returns first fall below the hurdle."""
        best = 0
        for i in range(1, len(frame)):
            if (
                frame.at[i, Col.BUDGET] > available
                or frame.at[i, Col.MARGINAL_RETURN] < self._hurdle
            ):
                break
            best = i
        budget, gain = frame.at[best, Col.BUDGET], frame.at[best, Col.GAIN]
        logger.info("Recommended budget", extra={"budget": round(budget), "gain": round(gain)})
        return best
