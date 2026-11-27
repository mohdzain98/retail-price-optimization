"""Tests for src.simulator.pricing: live re-optimization and scoring of regular prices.

With the pipeline's own settings the simulator must reproduce the stored
recommendations; scoring flags each broken guardrail.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import Config, Objective
from src.schema import Col, Metric, Scenario, kpi_col
from src.simulator import PriceSettings, PriceSimulator, SimulatorData

NEW_PRICE = kpi_col(Scenario.RECOMMENDED, Metric.PRICE)


@pytest.fixture(scope="module")
def simulator(processed_dir: Path) -> PriceSimulator:
    """Simulator over the synthetic recommendations."""
    return PriceSimulator(Config.load(), SimulatorData.load(processed_dir).recommendations)


@pytest.fixture(scope="module")
def defaults(simulator: PriceSimulator) -> PriceSettings:
    """The pipeline's own guardrails and margins."""
    return PriceSettings.from_config(Config.load(), simulator.categories)


class TestPriceSimulator:
    """PriceSimulator matches the pipeline and reacts to user settings."""

    @pytest.mark.parametrize("objective", [Objective.PROFIT, Objective.REVENUE])
    def test_reproduces_pipeline(
        self, processed_dir: Path, simulator: PriceSimulator, defaults: PriceSettings,
        objective: Objective,
    ) -> None:  # fmt: skip
        """Default settings give the stored recommended prices and break nothing."""
        recs = SimulatorData.load(processed_dir).recommendations
        stored = recs[recs[Col.OBJECTIVE] == objective.value].set_index(Col.UPC)[NEW_PRICE]
        result = simulator.optimize(objective, defaults)
        got = result.products.set_index(Col.UPC)[NEW_PRICE]
        np.testing.assert_allclose(got.loc[stored.index], stored, rtol=1e-6)
        assert result.broken == ()
        assert result.totals.loc[objective.value, "change"] >= -1e-9

    def test_score_current_prices(self, simulator: PriceSimulator, defaults: PriceSettings) -> None:
        """Current prices change nothing and break nothing."""
        result = simulator.score(
            Objective.PROFIT, defaults, simulator.products[Col.CURRENT_PRICE].to_numpy()
        )
        np.testing.assert_allclose(result.totals["change"], 0.0, atol=1e-12)
        assert result.broken == ()

    def test_score_flags_guardrails(
        self, simulator: PriceSimulator, defaults: PriceSettings
    ) -> None:
        """+25% everywhere breaks the bounds, the volume floor and the average cap."""
        prices = simulator.products[Col.CURRENT_PRICE].to_numpy() * 1.25
        broken = set(simulator.score(Objective.PROFIT, defaults, prices).broken)
        assert {"price_bounds", "volume_floor", "max_avg_price_increase"} <= broken

    def test_margins_set_unit_cost(self, simulator: PriceSimulator) -> None:
        """Unit cost = (1 - margin) x current price per category."""
        inputs = simulator.inputs({c: 0.5 for c in simulator.categories})
        np.testing.assert_allclose(inputs.cost, 0.5 * inputs.p0)

    def test_curve_peak(self, simulator: PriceSimulator, defaults: PriceSettings) -> None:
        """For an elastic product the profit curve peaks at c * beta / (1 + beta)."""
        upc = int(simulator.products[Col.UPC].iloc[0])
        curve = simulator.curve(Objective.PROFIT, defaults, upc)
        inputs = simulator.inputs(dict(defaults.margins))
        i = 0
        expected = inputs.cost[i] * inputs.beta[i] / (1 + inputs.beta[i])
        peak = curve.loc[curve[Col.OBJECTIVE_VALUE].idxmax(), Col.PRICE]
        step = curve[Col.PRICE].diff().iloc[1]
        if 0.7 * inputs.p0[i] < expected < 1.3 * inputs.p0[i]:
            assert abs(peak - expected) <= step
