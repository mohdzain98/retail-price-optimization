"""Tests for src.optimization.demand: values, slopes and the linear approximation gap.

Slopes are checked against finite differences; the linear curve must match the
constant-elasticity curve at r = 1 and diverge away from it.
One test class per production class.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.optimization.demand import ConstantElasticity, DemandCurve, LinearElasticity

R = np.array([0.8, 1.0, 1.2])
BETA = np.array([-2.0, -2.0, -2.0])
BASE = np.array([100.0, 100.0, 100.0])


def _numeric_slope(curve: DemandCurve, h: float = 1e-6) -> np.ndarray:
    """Central finite-difference dQ/dr."""
    return (curve.volume(R + h, BETA, BASE) - curve.volume(R - h, BETA, BASE)) / (2 * h)


class TestConstantElasticity:
    """Q = B r^beta."""

    def test_values_and_slope(self) -> None:
        """Known values at +/-20% and analytic slope matches finite differences."""
        curve = ConstantElasticity()
        np.testing.assert_allclose(
            curve.volume(R, BETA, BASE), [156.25, 100.0, 69.444444], rtol=1e-6
        )
        np.testing.assert_allclose(curve.slope(R, BETA, BASE), _numeric_slope(curve), rtol=1e-5)


class TestLinearElasticity:
    """Q = B (1 + beta (r - 1)), floored at zero."""

    def test_matches_at_current_and_diverges(self) -> None:
        """Equal at r=1; at +/-20% the linear curve gives 140 / 60 instead of 156 / 69."""
        linear, constant = LinearElasticity(), ConstantElasticity()
        np.testing.assert_allclose(linear.volume(R, BETA, BASE), [140.0, 100.0, 60.0])
        assert linear.volume(R, BETA, BASE)[1] == constant.volume(R, BETA, BASE)[1]
        np.testing.assert_allclose(linear.slope(R, BETA, BASE), _numeric_slope(linear), rtol=1e-5)

    def test_floor_at_zero(self) -> None:
        """Volume never goes negative for large price increases."""
        q = LinearElasticity().volume(np.array([2.0]), np.array([-3.0]), np.array([100.0]))
        assert q[0] == pytest.approx(0.0)
