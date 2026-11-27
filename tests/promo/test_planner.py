"""Tests for src.promo.planner: rules, the MILP and greedy planners, last-year replay.

The MILP must respect every rule and never score below greedy; a small knapsack
shows a case where greedy is strictly worse. One test class per production class.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.promo.planner import GreedyPlanner, LastYearCalendar, MilpPlanner, PlanRules
from src.schema import Col, Tactic
from tests.promo.conftest import make_candidates, random_candidates

RULES = PlanRules(max_promos=3, min_gap_weeks=2, feature_slots=2, display_slots=3)


class TestPlanRules:
    """PlanRules flags each kind of breach."""

    def test_valid_plan(self) -> None:
        """A plan inside every rule has no violations."""
        plan = make_candidates([(1, 1, "tpr", 5, 10), (1, 4, "tpr", 5, 10)])
        assert RULES.violations(plan, budget=20) == []

    def test_each_breach(self) -> None:
        """Budget, gap, promos per SKU and feature slots are all detected."""
        plan = make_candidates(
            [(1, 1, "feature", 5, 10), (1, 2, "tpr", 5, 10), (1, 5, "tpr", 5, 10),
             (1, 9, "tpr", 5, 10), (2, 1, "feature", 5, 10), (3, 1, "feature_display", 5, 10)]
        )  # fmt: skip
        assert set(RULES.violations(plan, budget=30)) == {
            "budget", "min_gap", "max_promos", "feature_slots"
        }  # fmt: skip

    def test_from_config(self) -> None:
        """Rules mirror the promo config; the window is gap + 1 weeks."""
        cfg = Config.load().promo
        rules = PlanRules.from_config(cfg)
        assert rules.max_promos == cfg.max_promos_per_sku
        assert rules.window == cfg.min_gap_weeks + 1


class TestMilpPlanner:
    """MilpPlanner returns the best rule-abiding plan."""

    def test_beats_greedy_on_knapsack(self) -> None:
        """Greedy takes the best-ROI item and strands budget; the MILP takes the pair."""
        cands = make_candidates([(1, 1, "tpr", 7.0, 6.0), (2, 1, "tpr", 5.5, 5.0),
                                 (3, 1, "tpr", 5.5, 5.0)])  # fmt: skip
        milp = MilpPlanner(RULES).plan(cands, budget=10)
        greedy = GreedyPlanner(RULES).plan(cands, budget=10)
        assert cands.loc[milp, Col.GAIN].sum() == pytest.approx(11.0)
        assert cands.loc[greedy, Col.GAIN].sum() == pytest.approx(7.0)

    @pytest.mark.parametrize("budget", [0.0, 5_000.0, 20_000.0, math.inf])
    def test_rules_and_dominance(self, budget: float) -> None:
        """No rule broken at any budget, and the MILP gain is at least greedy's."""
        cands = random_candidates()
        milp = cands[MilpPlanner(RULES, mip_gap=0).plan(cands, budget)]
        greedy = cands[GreedyPlanner(RULES).plan(cands, budget)]
        for plan in (milp, greedy):
            assert RULES.violations(plan, budget) == []
            assert (plan[Col.GAIN] > 0).all()
        assert milp[Col.GAIN].sum() >= greedy[Col.GAIN].sum() - 1e-6

    def test_gap_spreads_promos(self) -> None:
        """One SKU, every week worth promoting: 3 promos, at least 3 weeks apart."""
        cands = make_candidates([(1, w, "tpr", 10.0, 1.0) for w in range(1, 9)])
        weeks = np.sort(cands.loc[MilpPlanner(RULES).plan(cands, budget=100), Col.HORIZON_WEEK])
        assert len(weeks) == 3
        assert np.diff(weeks).min() >= RULES.window

    def test_feature_slots(self) -> None:
        """Three SKUs want a feature in week 1; only two circular slots exist."""
        cands = make_candidates([(u, 1, "feature", 10.0, 1.0) for u in (1, 2, 3)])
        assert MilpPlanner(RULES).plan(cands, budget=100).sum() == 2

    def test_nothing_worth_doing(self) -> None:
        """Only negative gains: the plan is empty."""
        cands = make_candidates([(1, 1, "tpr", -5.0, 1.0)])
        assert not MilpPlanner(RULES).plan(cands, budget=100).any()


class TestGreedyPlanner:
    """GreedyPlanner ranks by gain per dollar."""

    def test_takes_best_ratio_first(self) -> None:
        """With budget for one, it takes the higher-ROI option of a SKU-week."""
        cands = make_candidates([(1, 1, "tpr", 10.0, 10.0), (1, 1, "display", 12.0, 6.0)])
        plan = cands[GreedyPlanner(RULES).plan(cands, budget=10)]
        assert plan[Col.TACTIC].tolist() == ["display"]


class TestLastYearCalendar:
    """LastYearCalendar maps history to options and matches candidates."""

    @pytest.fixture
    def history(self) -> pd.DataFrame:
        """Weekly panel ending 2012-01-04 with three promos 52 weeks before weeks 1-2."""
        weeks = pd.date_range(end="2012-01-04", periods=60, freq="7D")
        frame = pd.DataFrame({Col.WEEK: weeks, Col.UPC: 1, Col.PROMO_FLAG: 0, Col.FEATURE: 0.0,
                              Col.DISPLAY: 0.0, Col.DISCOUNT_DEPTH: 0.0})  # fmt: skip
        week1 = weeks[-1] - pd.Timedelta(weeks=51)
        promos = pd.DataFrame(
            {Col.WEEK: [week1, week1, week1 + pd.Timedelta(weeks=1)], Col.UPC: [1, 2, 1],
             Col.PROMO_FLAG: 1, Col.FEATURE: [0.8, 0.0, 0.0], Col.DISPLAY: [0.6, 0.5, 0.0],
             Col.DISCOUNT_DEPTH: [0.32, 0.04, 0.17]}
        )  # fmt: skip
        return pd.concat([frame, promos], ignore_index=True)

    def test_keys(self, history: pd.DataFrame) -> None:
        """Tactic from merchandising shares, depth snapped to the nearest option."""
        keys = LastYearCalendar(0.25, [0.1, 0.2, 0.3]).keys(history, horizon=13)
        got = keys.sort_values([Col.HORIZON_WEEK, Col.UPC]).reset_index(drop=True)
        assert got[Col.HORIZON_WEEK].tolist() == [1, 1, 2]
        assert got[Col.TACTIC].tolist() == [Tactic.FEATURE_DISPLAY, Tactic.DISPLAY, Tactic.TPR]
        np.testing.assert_allclose(got[Col.DISCOUNT_DEPTH], [0.3, 0.1, 0.2])

    def test_mask(self, history: pd.DataFrame) -> None:
        """Candidates matching last year's keys are selected, others not."""
        replay = LastYearCalendar(0.25, [0.1, 0.2, 0.3])
        cands = make_candidates([(1, 2, "tpr", 1, 1), (1, 3, "tpr", 1, 1)])
        cands[Col.DISCOUNT_DEPTH] = 0.2
        mask = replay.mask(cands, replay.keys(history, horizon=13))
        assert mask.tolist() == [True, False]
