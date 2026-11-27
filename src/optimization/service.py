"""Facade and composition root for the optimization layer (Phase 4).

OptimizationService builds inputs, solves the portfolio problem with every configured
NLopt algorithm, rounds the primary solution to price points, scores the linear
formula on the true curve, measures robustness and writes the recommendations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import OPTIMIZATION_SUMMARY_PATH, RECOMMENDATIONS_PATH, Config, DemandForm
from src.optimization.demand import DEMAND_CURVES, DemandCurve
from src.optimization.inputs import InputsBuilder, OptimizationInputs, PackSizeParser
from src.optimization.problem import KpiCalculator, ProblemBuilder
from src.optimization.robustness import RobustnessAnalyzer
from src.optimization.solver import (
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
    """Per-product recommendations and one summary row per scenario/algorithm."""

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


class OptimizationService:
    """Facade: inputs -> solve (all algorithms) -> round -> linear check -> robustness -> write.

    KPIs are weekly; the summary also reports annualised deltas.
    """

    def __init__(
        self,
        parts: OptimizationParts,
        demand: DemandCurve,
        annualize_weeks: int,
        outputs: OptimizationOutputs,
    ) -> None:
        self._parts = parts
        self._demand = demand
        self._weeks = annualize_weeks
        self._outputs = outputs
        self._kpis = KpiCalculator()

    def run(
        self, baseline: pd.DataFrame, final: pd.DataFrame, draws: pd.DataFrame
    ) -> OptimizationReport:
        """Optimise, score and persist; returns the report."""
        inputs = self._parts.inputs.build(baseline, final)
        results = [
            s.solve(self._parts.problems.build(inputs, self._demand)) for s in self._parts.solvers
        ]
        ratios = self._scenario_ratios(inputs, results[0])
        table = self._recommendations(inputs, ratios)
        robust = self._parts.robustness.run(inputs, self._demand, draws)
        table = table.merge(robust, on=Col.UPC, how="left")
        summary = self._summary(inputs, ratios, results)
        report = OptimizationReport(table, summary)
        self._write(report)
        return report

    def _scenario_ratios(
        self, inputs: OptimizationInputs, primary: SolveResult
    ) -> dict[Scenario, np.ndarray]:
        """Relative prices per scenario: current, optimal, rounded (repaired), linear optimum."""
        problem = self._parts.problems.build(inputs, self._demand)
        lo, hi = (b * inputs.p0 for b in problem.bounds)
        rounded = self._parts.rounder.round(primary.ratios * inputs.p0, lo, hi)
        rounded = self._parts.repair.repair(problem, rounded)
        linear = DEMAND_CURVES.get(DemandForm.LINEAR)()
        linear_result = self._parts.solvers[0].solve(self._parts.problems.build(inputs, linear))
        return {
            Scenario.CURRENT: np.ones(len(inputs.p0)),
            Scenario.OPTIMAL: primary.ratios,
            Scenario.RECOMMENDED: rounded / inputs.p0,
            Scenario.LINEAR_FORMULA: linear_result.ratios,
        }

    def _recommendations(
        self, inputs: OptimizationInputs, ratios: dict[Scenario, np.ndarray]
    ) -> pd.DataFrame:
        """Inputs plus weekly KPIs per scenario, all scored on the configured demand curve."""
        columns = {}
        for scenario, r in ratios.items():
            for metric, values in self._kpis.evaluate(inputs, self._demand, r).items():
                columns[kpi_col(scenario, metric)] = values
        table = inputs.products.assign(**columns)
        optimal_price = table[kpi_col(Scenario.RECOMMENDED, Metric.PRICE)]
        return table.assign(
            **{
                Col.PRICE_RATIO: optimal_price / table[Col.CURRENT_PRICE] - 1,
                Col.LINEAR_FORMULA_PRICE: table[kpi_col(Scenario.LINEAR_FORMULA, Metric.PRICE)],
            }
        )

    def _summary(
        self,
        inputs: OptimizationInputs,
        ratios: dict[Scenario, np.ndarray],
        results: Sequence[SolveResult],
    ) -> pd.DataFrame:
        """One row per algorithm plus the rounded and linear-formula scenarios."""
        current = self._totals(inputs, ratios[Scenario.CURRENT])
        rows = [self._row(r.algorithm.value, inputs, r.ratios, current, r) for r in results] + [
            self._row(s.value, inputs, ratios[s], current, None)
            for s in (Scenario.RECOMMENDED, Scenario.LINEAR_FORMULA)
        ]
        summary = pd.DataFrame(rows)
        logger.info(
            "Optimization summary",
            extra={"rows": summary[[Col.SCENARIO, "profit_change"]].round(4).to_dict("records")},
        )
        return summary

    def _row(
        self, name: str, inputs: OptimizationInputs, r: np.ndarray, current: dict[Metric, float],
        result: SolveResult | None,
    ) -> dict:  # fmt: skip
        """Summary row: weekly totals, % change vs current, annual profit delta, solver stats."""
        totals = self._totals(inputs, r)
        row: dict[str, object] = {Col.SCENARIO: name}
        for metric in (Metric.VOLUME, Metric.REVENUE, Metric.PROFIT):
            row[f"weekly_{metric.value}"] = totals[metric]
            row[f"{metric.value}_change"] = totals[metric] / current[metric] - 1
        row["annual_profit_delta"] = (totals[Metric.PROFIT] - current[Metric.PROFIT]) * self._weeks
        row[Col.EVALUATIONS] = result.evaluations if result else np.nan
        row[Col.FIT_SECONDS] = result.seconds if result else np.nan
        row[Col.STATUS] = result.status if result else ""
        row[Col.MAX_VIOLATION] = result.max_violation if result else self._max_violation(inputs, r)
        return row

    def _totals(self, inputs: OptimizationInputs, r: np.ndarray) -> dict[Metric, float]:
        """Weekly portfolio totals at relative prices r."""
        return {m: float(v.sum()) for m, v in self._kpis.evaluate(inputs, self._demand, r).items()}

    def _max_violation(self, inputs: OptimizationInputs, r: np.ndarray) -> float:
        """Constraint violation of a non-solver scenario (e.g. after rounding)."""
        return self._parts.problems.build(inputs, self._demand).max_violation(r)

    def _write(self, report: OptimizationReport) -> None:
        """Persist both tables as parquet."""
        for frame, path in (
            (report.recommendations, self._outputs.recommendations),
            (report.summary, self._outputs.summary),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
        logger.info("Wrote recommendations", extra={"products": len(report.recommendations)})


class OptimizationFactory:
    """Builds an OptimizationService from config (demand, objective, constraints, solvers).

    The only place that instantiates optimization classes.
    """

    def __init__(self, cfg: Config, outputs: OptimizationOutputs) -> None:
        self._cfg = cfg
        self._outputs = outputs

    def build(self) -> OptimizationService:
        """Wire inputs, problem builder, solvers, rounder and robustness analysis."""
        opt = self._cfg.optimization
        solvers = [NloptSolver(a, opt.maxeval, opt.xtol_rel) for a in opt.algorithms]
        problems = ProblemBuilder(opt)
        rounder = PricePointRounder(opt.price_endings)
        parts = OptimizationParts(
            inputs=InputsBuilder(opt.reference_weeks, self._cfg.cost, PackSizeParser()),
            problems=problems,
            solvers=solvers,
            rounder=rounder,
            repair=FeasibilityRepair(rounder),
            robustness=RobustnessAnalyzer(problems, solvers[0], opt.robustness_draws, opt.seed),
        )
        demand = DEMAND_CURVES.get(opt.demand_form)()
        return OptimizationService(parts, demand, opt.annualize_weeks, self._outputs)
