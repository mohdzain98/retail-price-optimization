"""NLopt solver wrapper and price-point rounding.

NloptSolver maximises a PricingProblem over relative prices with any configured
NLopt algorithm, passing analytic gradients to gradient-based ones and vector
constraints in fc(r) <= 0 form. PricePointRounder snaps prices to endings like .99.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import nlopt
import numpy as np

from src.config import Algorithm
from src.optimization.problem import Constraint, PricingProblem
from src.utils import AppLogger

logger = AppLogger.get_logger(__name__)

CONSTRAINT_TOL = 1e-8


@dataclass(frozen=True)
class SolveResult:
    """Outcome of one optimization run."""

    algorithm: Algorithm
    ratios: np.ndarray
    objective: float
    evaluations: int
    seconds: float
    status: str
    max_violation: float


class _Tracker:
    """Counts objective calls and keeps the best feasible point seen."""

    def __init__(self, problem: PricingProblem) -> None:
        self.problem = problem
        self.calls = 0
        self.best_x: np.ndarray | None = None
        self.best_value = -np.inf

    def record(self, x: np.ndarray, value: float) -> None:
        """Update the best point if x is feasible and better."""
        self.calls += 1
        if value > self.best_value and self.problem.max_violation(x) <= CONSTRAINT_TOL * 10:
            self.best_x, self.best_value = x.copy(), value


class NloptSolver:
    """Maximises the problem objective with one NLopt algorithm, starting from current prices.

    Returns NLopt's optimum, or the best feasible point seen if NLopt stops early.
    """

    def __init__(self, algorithm: Algorithm, maxeval: int, xtol_rel: float) -> None:
        self._algorithm = algorithm
        self._maxeval = maxeval
        self._xtol_rel = xtol_rel

    @property
    def algorithm(self) -> Algorithm:
        """The NLopt algorithm used."""
        return self._algorithm

    def solve(self, problem: PricingProblem) -> SolveResult:
        """Run NLopt and return the optimum in relative prices."""
        tracker = _Tracker(problem)
        opt = self._configure(problem, tracker)
        start = time.perf_counter()
        x0 = np.ones(problem.n)  # current prices are always feasible
        try:
            x = np.asarray(opt.optimize(x0))
            status = _STATUS.get(opt.last_optimize_result(), "unknown")
        except (nlopt.RoundoffLimited, nlopt.ForcedStop, RuntimeError) as err:
            logger.warning(
                "NLopt stopped early", extra={"algorithm": self._algorithm.value, "error": str(err)}
            )
            x, status = (tracker.best_x if tracker.best_x is not None else x0), "stopped_early"
        value, _ = problem.objective.value_and_grad(x, problem.inputs, problem.demand)
        return SolveResult(
            algorithm=self._algorithm,
            ratios=x,
            objective=value,
            evaluations=tracker.calls,
            seconds=round(time.perf_counter() - start, 3),
            status=status,
            max_violation=problem.max_violation(x),
        )

    def _configure(self, problem: PricingProblem, tracker: _Tracker) -> nlopt.opt:
        """NLopt object with bounds, objective, constraints and stopping rules.

        The objective is divided by its value at current prices so it is O(1).
        """
        opt = nlopt.opt(getattr(nlopt, self._algorithm.value), problem.n)
        at_current, _ = problem.objective.value_and_grad(
            np.ones(problem.n), problem.inputs, problem.demand
        )
        scale = max(abs(at_current), 1e-12)
        lo, hi = problem.bounds
        opt.set_lower_bounds(np.full(problem.n, lo))
        opt.set_upper_bounds(np.full(problem.n, hi))

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            value, gradient = problem.objective.value_and_grad(x, problem.inputs, problem.demand)
            tracker.record(x, value)
            if grad.size:
                grad[:] = -gradient / scale  # NLopt minimises
            return -value / scale

        opt.set_min_objective(objective)
        for constraint in problem.constraints:
            if constraint.size:
                opt.add_inequality_mconstraint(
                    self._constraint_fn(constraint, problem), [CONSTRAINT_TOL] * constraint.size
                )
        opt.set_xtol_rel(self._xtol_rel)
        opt.set_ftol_rel(self._xtol_rel)
        opt.set_maxeval(self._maxeval)
        return opt

    @staticmethod
    def _constraint_fn(
        constraint: Constraint, problem: PricingProblem
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], None]:
        """NLopt mconstraint callback: fills result (m) and grad (m x n)."""

        def fn(result: np.ndarray, x: np.ndarray, grad: np.ndarray) -> None:
            values, jac = constraint.evaluate(x, problem.inputs, problem.demand)
            result[:] = values
            if grad.size:
                grad[:] = jac

        return fn


_STATUS: dict[int, str] = {
    nlopt.SUCCESS: "success",
    nlopt.STOPVAL_REACHED: "stopval_reached",
    nlopt.FTOL_REACHED: "ftol_reached",
    nlopt.XTOL_REACHED: "xtol_reached",
    nlopt.MAXEVAL_REACHED: "maxeval_reached",
    nlopt.MAXTIME_REACHED: "maxtime_reached",
}


class PricePointRounder:
    """Snaps prices to the nearest allowed ending (e.g. x.09 ... x.99) inside the bounds.

    If the nearest ending falls outside [low, high], the nearest ending inside is used.
    """

    def __init__(self, endings: Sequence[float]) -> None:
        self._endings = np.asarray(sorted(endings))

    def round(self, prices: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        """Rounded prices, one per product."""
        return np.array([self._one(p, lo, hi) for p, lo, hi in zip(prices, low, high, strict=True)])

    def step_toward(self, price: float, target: float) -> float:
        """Next price point from price toward target (target itself if no point lies between)."""
        between = [
            c for c in self._candidates(price) if min(price, target) < c < max(price, target)
        ]
        if not between:
            return float(target)
        return float(min(between, key=lambda c: abs(c - price)))

    def _one(self, price: float, low: float, high: float) -> float:
        """Nearest candidate ending to price within [low, high] (price itself if none fits)."""
        candidates = self._candidates(price)
        inside = candidates[(candidates >= low) & (candidates <= high)]
        if inside.size == 0:
            return float(price)
        return float(inside[np.argmin(np.abs(inside - price))])

    def _candidates(self, price: float) -> np.ndarray:
        """Price points in the dollar bands around price."""
        dollars = np.floor(price)
        return np.concatenate([d + self._endings for d in (dollars - 1, dollars, dollars + 1)])


class FeasibilityRepair:
    """Restores feasibility after rounding by stepping prices back toward current prices.

    Greedy: each step moves the one product whose move most reduces the worst violation.
    """

    def __init__(self, rounder: PricePointRounder, max_steps: int = 500) -> None:
        self._rounder = rounder
        self._max_steps = max_steps

    def repair(self, problem: PricingProblem, prices: np.ndarray) -> np.ndarray:
        """Feasible prices (or the best found, with a warning, if steps run out)."""
        p0 = problem.inputs.p0
        prices = prices.copy()
        for _ in range(self._max_steps):
            violation = problem.max_violation(prices / p0)
            movable = np.flatnonzero(~np.isclose(prices, p0))
            if violation <= CONSTRAINT_TOL * 10 or movable.size == 0:
                return prices
            trials = [self._trial(problem, prices, int(i)) for i in movable]
            best = int(np.argmin([v for v, _ in trials]))
            prices = trials[best][1]
        logger.warning("Rounding repair did not reach feasibility", extra={"violation": violation})
        return prices

    def _trial(
        self, problem: PricingProblem, prices: np.ndarray, i: int
    ) -> tuple[float, np.ndarray]:
        """Violation after stepping product i one price point toward its current price."""
        p0 = problem.inputs.p0
        candidate = prices.copy()
        candidate[i] = self._rounder.step_toward(prices[i], p0[i])
        return problem.max_violation(candidate / p0), candidate
