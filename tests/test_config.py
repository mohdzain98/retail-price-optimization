"""Tests for the configuration loader.

Ensures the shipped config.yaml validates and that guard-rail validators
reject economically invalid settings (e.g. positive elasticity bounds).
Runs without the raw dataset so it is safe in CI.
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from src.config import CONFIG_PATH, Config, load_config


def _raw() -> dict:
    """Return the shipped config as a plain dict for mutation in tests."""
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_shipped_config_loads() -> None:
    """The committed config.yaml validates and exposes expected defaults."""
    cfg = load_config()
    assert cfg.optimization.objective == "profit"
    assert cfg.elasticity.bounds[1] < 0
    assert cfg.data.raw_path.parent.name == "raw"


def test_margin_falls_back_to_default() -> None:
    """Unknown categories use the default margin."""
    cfg = load_config()
    assert cfg.cost.margin_for("UNKNOWN") == cfg.cost.default_margin


@pytest.mark.parametrize("bounds", [[-0.3, -5.0], [-1.0, 0.5]])
def test_rejects_bad_elasticity_bounds(bounds: list[float]) -> None:
    """Unordered or non-negative elasticity bounds fail validation."""
    raw = _raw()
    raw["elasticity"]["bounds"] = bounds
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_rejects_bad_price_bounds() -> None:
    """Price bounds must bracket 1.0 (the current price)."""
    raw = _raw()
    raw["optimization"]["price_bounds"] = [1.1, 1.2]
    with pytest.raises(ValidationError):
        Config.model_validate(raw)
