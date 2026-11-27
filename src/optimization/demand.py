"""Demand curves used inside the optimizer, in relative price r = P / P0.

ConstantElasticity (Q = B r^beta) is consistent with the log-log elasticity model.
LinearElasticity (Q = B (1 + beta (r - 1))) is the first-order approximation used by
the RGMx notebook and the article; it is kept to show how far its optimum drifts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from src.config import DemandForm
from src.utils import Registry


class DemandCurve(ABC):
    """Weekly volume and its derivative with respect to relative price.

    Both are vectorised over products: r, beta and base have one entry per product.
    """

    form: ClassVar[DemandForm]

    @abstractmethod
    def volume(self, r: np.ndarray, beta: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Weekly volume at relative price r."""

    @abstractmethod
    def slope(self, r: np.ndarray, beta: np.ndarray, base: np.ndarray) -> np.ndarray:
        """dQ/dr, used for analytic gradients."""


DEMAND_CURVES: Registry[DemandForm, type[DemandCurve]] = Registry("form")


@DEMAND_CURVES.register
class ConstantElasticity(DemandCurve):
    """Q = B r^beta: the curve implied by a log-log model (elasticity constant everywhere)."""

    form = DemandForm.CONSTANT_ELASTICITY

    def volume(self, r: np.ndarray, beta: np.ndarray, base: np.ndarray) -> np.ndarray:
        """B r^beta."""
        return base * np.power(r, beta)

    def slope(self, r: np.ndarray, beta: np.ndarray, base: np.ndarray) -> np.ndarray:
        """beta B r^(beta - 1)."""
        return beta * base * np.power(r, beta - 1)


@DEMAND_CURVES.register
class LinearElasticity(DemandCurve):
    """Q = B (1 + beta (r - 1)), floored at zero: the linear (Taylor) approximation.

    Accurate only near r = 1; over +/-20% it misstates volume noticeably.
    """

    form = DemandForm.LINEAR

    def volume(self, r: np.ndarray, beta: np.ndarray, base: np.ndarray) -> np.ndarray:
        """B (1 + beta (r - 1)), never negative."""
        return np.maximum(base * (1 + beta * (r - 1)), 0.0)

    def slope(self, r: np.ndarray, beta: np.ndarray, base: np.ndarray) -> np.ndarray:
        """B beta where volume is positive, else 0."""
        return np.where(1 + beta * (r - 1) > 0, base * beta, 0.0)
