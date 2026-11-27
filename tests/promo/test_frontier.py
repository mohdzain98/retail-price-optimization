"""Tests for src.promo.frontier: the budget frontier and the recommended budget.

Gain must rise with budget, the recommendation must stay within the available
budget, and a prohibitive hurdle must recommend spending nothing.
"""

from __future__ import annotations

import numpy as np

from src.promo.frontier import BudgetFrontier
from src.promo.planner import MilpPlanner, PlanRules
from src.schema import Col
from tests.promo.conftest import random_candidates

RULES = PlanRules(max_promos=4, min_gap_weeks=3, feature_slots=3, display_slots=3)


class TestBudgetFrontier:
    """BudgetFrontier traces gain against budget."""

    def test_shape_and_recommendation(self) -> None:
        """Starts at 0, gain non-decreasing, one recommended row within the available budget."""
        cands = random_candidates(seed=1)
        frame = BudgetFrontier(MilpPlanner(RULES), 6, 0.0).trace(cands, available=8_000)
        assert frame[Col.BUDGET].iloc[0] == 0 and frame[Col.GAIN].iloc[0] == 0
        assert (np.diff(frame[Col.GAIN]) >= -1e-6).all()
        assert (frame[Col.TRADE_SPEND] <= frame[Col.BUDGET] + 1e-6).all()
        assert 8_000 in frame[Col.BUDGET].to_numpy()
        chosen = frame[frame[Col.RECOMMENDED_BUDGET]]
        assert len(chosen) == 1 and chosen[Col.BUDGET].iloc[0] <= 8_000

    def test_prohibitive_hurdle(self) -> None:
        """No step returns 1000x its cost, so the recommendation is to spend nothing."""
        frame = BudgetFrontier(MilpPlanner(RULES), 4, 1_000.0).trace(
            random_candidates(), available=1e9
        )
        assert frame.loc[frame[Col.RECOMMENDED_BUDGET], Col.BUDGET].iloc[0] == 0
