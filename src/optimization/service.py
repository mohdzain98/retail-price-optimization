"""Facade and composition root for the optimization layer (Phase 4).

For every configured objective (e.g. profit and revenue) OptimizationService solves
the portfolio with each NLopt algorithm, rounds to price points with a feasibility
repair and a never-worse-than-current safeguard, scores the linear formula, checks
the pricing-direction rule and robustness, and writes both side by side.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    OPTIMIZATION_SUMMARY_PATH,
    RECOMMENDATIONS_PATH,
    Config,
    DemandForm,
    Objective,
)
from src.optimization.checks import DirectionRuleCheck
from src.optimization.demand import DEMAND_CURVES, DemandCurve
from src.optimization.inputs import InputsBuilder, OptimizationInputs, PackSizeParser
from src.optimization.problem import KpiCalculator, PricingProblem, ProblemBuilder
from src.optimization.robustness import RobustnessAnalyzer
from src.optimization.solver import (
    CONSTRAINT_TOL,
    FeasibilityRepair,
    NloptSolver,
    PricePointRounder,
    SolveResult,
)
from src.schema import Col, Metric, Scenario, kpi_col
from src.utils import AppLogger

logger = AppLogger.get_logger(__name__)


@dataclass(frozen=True)
class OptimizationOutputs:
    """Where the stage writes per-product recommendations and the run summary."""

    recommendations: Path
    summary: Path

    @classmethod
    def default(cls) -> OptimizationOutputs:
        """Standard locations under data/processed."""
        return cls(RECOMMENDATIONS_PATH, OPTIMIZATION_SUMMARY_PATH)


@dataclass(frozen=True)
class OptimizationReport:
    """Per-product recommendations and summary rows, one block per objective."""

    recommendations: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class OptimizationParts:
    """Collaborators of the service, grouped to keep the constructor small."""

    inputs: InputsBuilder
    problems: ProblemBuilder
    solvers: Sequence[NloptSolver]  # first is primary
    rounder: PricePointRounder
    repair: FeasibilityRepair
    robustness: RobustnessAnalyzer
    rules: DirectionRuleCheck


@dataclass(frozen=True)
class _Recommended:
    """Rounded relative prices and whether the safeguard reverted them to current."""

    ratios: np.ndarray
    safeguard_applied: bool


class OptimizationService:
    """Facade: per objective, solve -> round -> safeguard -> checks -> robustness; then write.

    KPIs are weekly and always report real profit (unit cost), whatever the objective.
    """

    def __init__(
        self,
        parts: OptimizationParts,
        demand: DemandCurve,
        objectives: Sequence[Objective],
        annualize_weeks: int,
        outputs: OptimizationOutputs,
    ) -> None:
        self._parts = parts
        self._demand = demand
        self._objectives = list(objectives)
        self._weeks = annualize_weeks
        self._outputs = outputs
        self._kpis = KpiCalculator()

    def run(
        self, baseline: pd.DataFrame, final: pd.DataFrame, draws: pd.DataFrame
    ) -> OptimizationReport:
        """Optimise every objective, stack the results and persist them."""
        inputs = self._parts.inputs.build(baseline, final)
        tables, summaries = [], []
        for objective in self._objectives:
            table, summary = self._solve(inputs, objective, draws)
            tables.append(table.assign(**{Col.OBJECTIVE: objective.value}))
            summaries.append(summary.assign(**{Col.OBJECTIVE: objective.value}))
        report = OptimizationReport(
            pd.concat(tables, ignore_index=True), pd.concat(summaries, ignore_index=True)
        )
        self._write(report)
        return report

    def _solve(
        self, inputs: OptimizationInputs, objective: Objective, draws: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Recommendations table and summary rows for one objective."""
        logger.info("Optimizing", extra={"objective": objective.value})
        problem = self._parts.problems.build(inputs, self._demand, objective)
        results = [solver.solve(problem) for solver in self._parts.solvers]
        recommended = self._recommended(problem, results[0].ratios)
        linear = DEMAND_CURVES.get(DemandForm.LINEAR)()
        linear_problem = self._parts.problems.build(inputs, linear, objective)
        ratios = {
            Scenario.CURRENT: np.ones(problem.n),
            Scenario.OPTIMAL: results[0].ratios,
            Scenario.RECOMMENDED: recommended.ratios,
            Scenario.LINEAR_FORMULA: self._parts.solvers[0].solve(linear_problem).ratios,
        }
        robust = self._parts.robustness.run(inputs, self._demand, draws, objective)
        table = (
            self._recommendations(inputs, ratios, recommended.safeguard_applied)
            .merge(self._parts.rules.check(problem, recommended.ratios), on=Col.UPC)
            .merge(robust, on=Col.UPC, how="left")
        )
        return table, self._summary(problem, ratios, results)

    def _recommended(self, problem: PricingProblem, optimal: np.ndarray) -> _Recommended:
        """Round to price points, repair feasibility, never score below current prices.

        Falls back to current prices (and logs a warning) if rounding loses the gain.
        """
        p0, inputs, demand = problem.inputs.p0, problem.inputs, problem.demand
        lo, hi = (b * p0 for b in problem.bounds)
        prices = self._parts.repair.repair(problem, self._parts.rounder.round(optimal * p0, lo, hi))
        ratios = prices / p0
        value, _ = problem.objective.value_and_grad(ratios, inputs, demand)
        current, _ = problem.objective.value_and_grad(np.ones(problem.n), inputs, demand)
        if value < current or problem.max_violation(ratios) > CONSTRAINT_TOL * 10:
            logger.warning("Rounded prices score below current; keeping current prices")
            return _Recommended(np.ones(problem.n), safeguard_applied=True)
        return _Recommended(ratios, safeguard_applied=False)

    def _recommendations(
        self, inputs: OptimizationInputs, ratios: dict[Scenario, np.ndarray], safeguard: bool
    ) -> pd.DataFrame:
        """Inputs plus weekly KPIs per scenario, all scored on the configured demand curve."""
        columns = {}
        for scenario, r in ratios.items():
            for metric, values in self._kpis.evaluate(inputs, self._demand, r).items():
                columns[kpi_col(scenario, metric)] = values
        table = inputs.products.assign(**columns)
        recommended_price = table[kpi_col(Scenario.RECOMMENDED, Metric.PRICE)]
        return table.assign(
            **{
                Col.PRICE_RATIO: recommended_price / table[Col.CURRENT_PRICE] - 1,
                Col.LINEAR_FORMULA_PRICE: table[kpi_col(Scenario.LINEAR_FORMULA, Metric.PRICE)],
                Col.SAFEGUARD_APPLIED: safeguard,
            }
        )

    def _summary(
        self,
        problem: PricingProblem,
        ratios: dict[Scenario, np.ndarray],
        results: Sequence[SolveResult],
    ) -> pd.DataFrame:
        """One row per algorithm plus the rounded and linear-formula scenarios."""
        current = self._totals(problem.inputs, ratios[Scenario.CURRENT])
        rows = [self._row(r.algorithm.value, problem, r.ratios, current, r) for r in results] + [
            self._row(s.value, problem, ratios[s], current, None)
            for s in (Scenario.RECOMMENDED, Scenario.LINEAR_FORMULA)
        ]
        return pd.DataFrame(rows)

    def _row(
        self, name: str, problem: PricingProblem, r: np.ndarray, current: dict[Metric, float],
        result: SolveResult | None,
    ) -> dict:  # fmt: skip
        """Summary row: weekly totals, % change vs current, annual profit delta, solver stats."""
        totals = self._totals(problem.inputs, r)
        row: dict[str, object] = {Col.SCENARIO: name}
        for metric in (Metric.VOLUME, Metric.REVENUE, Metric.PROFIT):
            row[f"weekly_{metric.value}"] = totals[metric]
            row[f"{metric.value}_change"] = totals[metric] / current[metric] - 1
        row["annual_profit_delta"] = (totals[Metric.PROFIT] - current[Metric.PROFIT]) * self._weeks
        row[Col.EVALUATIONS] = result.evaluations if result else np.nan
        row[Col.FIT_SECONDS] = result.seconds if result else np.nan
        row[Col.STATUS] = result.status if result else ""
        row[Col.MAX_VIOLATION] = result.max_violation if result else problem.max_violation(r)
        return row

    def _totals(self, inputs: OptimizationInputs, r: np.ndarray) -> dict[Metric, float]:
        """Weekly portfolio totals at relative prices r."""
        return {m: float(v.sum()) for m, v in self._kpis.evaluate(inputs, self._demand, r).items()}

    def _write(self, report: OptimizationReport) -> None:
        """Persist both tables as parquet."""
        for frame, path in (
            (report.recommendations, self._outputs.recommendations),
            (report.summary, self._outputs.summary),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
        logger.info("Wrote recommendations", extra={"rows": len(report.recommendations)})


class OptimizationFactory:
    """Builds an OptimizationService from config (demand, objectives, constraints, solvers).

    The only place that instantiates optimization classes.
    """

    def __init__(self, cfg: Config, outputs: OptimizationOutputs) -> None:
        self._cfg = cfg
        self._outputs = outputs

    def build(self) -> OptimizationService:
        """Wire inputs, problem builder, solvers, rounding, checks and robustness."""
        opt = self._cfg.optimization
        solvers = [
            NloptSolver(a, opt.maxeval, opt.xtol_rel, opt.max_seconds) for a in opt.algorithms
        ]
        problems = ProblemBuilder(opt)
        rounder = PricePointRounder(opt.price_endings)
        parts = OptimizationParts(
            inputs=InputsBuilder(opt.reference_weeks, self._cfg.cost, PackSizeParser()),
            problems=problems,
            solvers=solvers,
            rounder=rounder,
            repair=FeasibilityRepair(rounder),
            robustness=RobustnessAnalyzer(problems, solvers[0], opt.robustness_draws, opt.seed),
            rules=DirectionRuleCheck(),
        )
        demand = DEMAND_CURVES.get(opt.demand_form)()
        return OptimizationService(
            parts, demand, opt.objectives, opt.annualize_weeks, self._outputs
        )
