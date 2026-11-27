"""Tests for src.simulator.promo: live MILP re-planning from the stored options.

At the pipeline's settings and budget the simulator must match the stored MILP
plan; fee changes must re-price options consistently.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from src.config import Config
from src.promo.planner import PlanRules
from src.schema import Col, PromoScenario
from src.simulator import PromoSettings, PromoSimulator, SimulatorData
from src.simulator.promo import YOUR_PLAN


@pytest.fixture(scope="module")
def data(processed_dir: Path) -> SimulatorData:
    """Synthetic processed outputs."""
    return SimulatorData.load(processed_dir)


@pytest.fixture(scope="module")
def defaults() -> PromoSettings:
    """The pipeline's own promo settings."""
    return PromoSettings.from_config(Config.load())


class TestPromoSimulator:
    """PromoSimulator re-plans, re-prices and compares scenarios."""

    @pytest.mark.parametrize("objective", ["profit", "revenue"])
    def test_reproduces_pipeline(
        self, data: SimulatorData, defaults: PromoSettings, objective: str
    ) -> None:
        """At the stored plan's budget the gain matches the stored MILP plan."""
        plans, frontier = data.plans, data.frontier
        stored = plans[
            (plans[Col.OBJECTIVE] == objective)
            & (plans[Col.SCENARIO] == PromoScenario.MILP_PLAN.value)
        ]
        best = frontier[(frontier[Col.OBJECTIVE] == objective) & frontier[Col.RECOMMENDED_BUDGET]]
        budget = float(best[Col.BUDGET].iloc[0])
        result = PromoSimulator(data).plan(objective, dataclasses.replace(defaults, budget=budget))
        assert result.plan[Col.GAIN].sum() == pytest.approx(stored[Col.GAIN].sum(), rel=2e-3)
        assert PlanRules.violations(defaults.rules, result.plan, budget) == []

    def test_scenarios(self, data: SimulatorData, defaults: PromoSettings) -> None:
        """Summary has no promo, last year and the user's plan; zero budget adds nothing."""
        result = PromoSimulator(data).plan("profit", dataclasses.replace(defaults, budget=0.0))
        summary = result.summary.set_index(Col.SCENARIO)
        assert set(summary.index) == {
            PromoScenario.CURRENT_NO_PROMO.value, PromoScenario.RECOMMENDED_NO_PROMO.value,
            PromoScenario.LAST_YEAR_PLAN.value, YOUR_PLAN,
        }  # fmt: skip
        assert result.plan.empty
        regular = summary.loc[PromoScenario.RECOMMENDED_NO_PROMO.value]
        assert summary.loc[YOUR_PLAN, Col.PROFIT] == pytest.approx(regular[Col.PROFIT])

    def test_refee(self, data: SimulatorData, defaults: PromoSettings) -> None:
        """+100 on the display fee: spend +100, profit -100, profit-goal gain -100."""
        sim = PromoSimulator(data)
        fees = tuple((t, f + 100.0 if t == "display" else f) for t, f in defaults.fees)
        before = sim.candidates("profit", defaults)
        after = sim.candidates("profit", dataclasses.replace(defaults, fees=fees))
        display = (before[Col.TACTIC] == "display").to_numpy()
        np.testing.assert_allclose(
            (after[Col.TRADE_SPEND] - before[Col.TRADE_SPEND])[display], 100.0
        )
        np.testing.assert_allclose((after[Col.GAIN] - before[Col.GAIN])[display], -100.0)
        np.testing.assert_allclose((after[Col.GAIN] - before[Col.GAIN])[~display], 0.0)

    def test_revenue_gain_ignores_fees(self, data: SimulatorData, defaults: PromoSettings) -> None:
        """The revenue goal charges no fees, so its gain does not move."""
        sim = PromoSimulator(data)
        fees = tuple((t, f + 500.0) for t, f in defaults.fees)
        before = sim.candidates("revenue", defaults)
        after = sim.candidates("revenue", dataclasses.replace(defaults, fees=fees))
        np.testing.assert_allclose(after[Col.GAIN], before[Col.GAIN])

    def test_option_filters(self, data: SimulatorData, defaults: PromoSettings) -> None:
        """Only the chosen tactics and depths remain."""
        settings = dataclasses.replace(defaults, tactics=("tpr",), depths=(0.2,))
        options = PromoSimulator(data).candidates("profit", settings)
        assert set(options[Col.TACTIC]) == {"tpr"}
        np.testing.assert_allclose(options[Col.DISCOUNT_DEPTH], 0.2)

    def test_frontier(self, data: SimulatorData, defaults: PromoSettings) -> None:
        """One recommended budget, gain non-decreasing in budget."""
        frame = PromoSimulator(data).frontier("revenue", defaults, 4, 0.25)
        assert frame[Col.RECOMMENDED_BUDGET].sum() == 1
        assert (np.diff(frame[Col.GAIN]) >= -1e-6).all()
