"""Tests for src.config: the shipped YAML validates and guard-rails reject bad values.

Runs without the raw dataset so it is safe in CI. Covers economic sanity checks
(negative elasticity bounds, price bounds around 1.0), immutability and that
every Sheet has a name and a column mapping.
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from src.config import CONFIG_PATH, Config, Grain, Objective, Sheet
from src.schema import Col


def _raw() -> dict:
    """Shipped config as a plain dict for mutation in tests."""
    return yaml.safe_load(CONFIG_PATH.read_text())


class TestConfig:
    """Config.load validates, caches and enforces guard-rails."""

    def test_shipped_config_loads(self) -> None:
        """The committed config validates into enums and canonical columns."""
        cfg = Config.load()
        assert cfg.optimization.objective is Objective.PROFIT
        assert cfg.data.grain is Grain.PRODUCT_WEEK
        assert cfg.data.columns[Sheet.TRANSACTIONS]["UNITS"] is Col.UNITS
        assert Config.load() is cfg  # cached

    def test_is_frozen(self) -> None:
        """Config objects cannot be mutated."""
        with pytest.raises(ValidationError):
            Config.load().split.holdout_weeks = 1  # type: ignore[misc]

    def test_margin_falls_back_to_default(self) -> None:
        """Unknown categories use the default margin."""
        cost = Config.load().cost
        assert cost.margin_for("UNKNOWN") == cost.default_margin

    @pytest.mark.parametrize("bounds", [[-0.3, -5.0], [-1.0, 0.5]])
    def test_rejects_bad_elasticity_bounds(self, bounds: list[float]) -> None:
        """Unordered or non-negative elasticity bounds fail validation."""
        raw = _raw()
        raw["elasticity"]["bounds"] = bounds
        with pytest.raises(ValidationError):
            Config.model_validate(raw)

    def test_rejects_bad_price_bounds(self) -> None:
        """Price bounds must bracket 1.0 (the current price)."""
        raw = _raw()
        raw["optimization"]["price_bounds"] = [1.1, 1.2]
        with pytest.raises(ValidationError):
            Config.model_validate(raw)

    def test_rejects_missing_sheet_mapping(self) -> None:
        """Every Sheet needs a column mapping."""
        raw = _raw()
        del raw["data"]["columns"]["stores"]
        with pytest.raises(ValidationError, match="stores"):
            Config.model_validate(raw)

    def test_rejects_priority_model_not_fitted(self) -> None:
        """Every model in model_priority must be fitted."""
        raw = _raw()
        raw["elasticity"]["models"] = ["ols_per_product"]
        with pytest.raises(ValidationError, match="model_priority"):
            Config.model_validate(raw)

    def test_rejects_unknown_canonical_column(self) -> None:
        """Mapping a raw column to a non-Col name fails."""
        raw = _raw()
        raw["data"]["columns"]["products"]["UPC"] = "not_a_column"
        with pytest.raises(ValidationError):
            Config.model_validate(raw)
