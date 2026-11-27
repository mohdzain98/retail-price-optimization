"""Facade and composition root for the forecast and promo layer (Phase 5).

PromoService backtests the baseline forecasters and forecasts with the best one, then
per objective scores all promo options at the recommended prices, traces the budget
frontier, plans with the MILP and benchmarks, and writes the tables the app reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BUDGET_FRONTIER_PATH,
    FORECAST_BACKTEST_PATH,
    FORECAST_PATH,
    PROMO_CANDIDATES_PATH,
    PROMO_PLAN_PATH,
    PROMO_SUMMARY_PATH,
    Config,
    ModelName,
    Objective,
)
from src.elasticity import ElasticityModel
from src.prep.features import FourierSeasonality, HolidayFlags
from src.promo.candidates import (
    CandidateScorer,
    EffectEstimator,
    ExecutionLevels,
    PromoEffects,
)
from src.promo.forecast import BaselineForecaster, ForecastBacktest, FutureCalendar
from src.promo.forecasters import FORECASTERS, ForecastContext, SeasonalIndex
from src.promo.frontier import BudgetFrontier
from src.promo.planner import PLANNERS, LastYearCalendar, PlanRules, PromoPlanner
from src.schema import Col, Metric, Planner, PromoScenario, Scenario, kpi_col
from src.utils import AppLogger

logger = AppLogger.get_logger(__name__)

PLAN_COLS: tuple[Col, ...] = (
    Col.TACTIC, Col.DISCOUNT_DEPTH, Col.PROMO_PRICE, Col.PROMO_UNITS, Col.TRADE_SPEND,
    Col.INCREMENTAL, Col.INCREMENTAL_REVENUE, Col.INCREMENTAL_PROFIT,
)  # fmt: skip
PLAN_SCENARIOS: dict[Planner, PromoScenario] = {
    Planner.GREEDY: PromoScenario.GREEDY_PLAN,
    Planner.MILP: PromoScenario.MILP_PLAN,
}
TOTALS: tuple[Col, ...] = (Col.VOLUME, Col.REVENUE, Col.PROFIT)


@dataclass(frozen=True)
class PromoOutputs:
    """Where the stage writes forecast, backtest, options, plans, frontier and summary."""

    forecast: Path
    backtest: Path
    candidates: Path
    plan: Path
    frontier: Path
    summary: Path

    @classmethod
    def default(cls) -> PromoOutputs:
        """Standard locations under data/processed."""
        return cls(FORECAST_PATH, FORECAST_BACKTEST_PATH, PROMO_CANDIDATES_PATH,
                   PROMO_PLAN_PATH, BUDGET_FRONTIER_PATH, PROMO_SUMMARY_PATH)  # fmt: skip


@dataclass(frozen=True)
class PromoReport:
    """All Phase 5 tables; every one except the backtest has an objective column."""

    forecast: pd.DataFrame
    backtest: pd.DataFrame
    candidates: pd.DataFrame
    plan: pd.DataFrame
    frontier: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class PromoParts:
    """Collaborators of the service, grouped to keep the constructor small."""

    calendar: FutureCalendar
    forecaster: BaselineForecaster
    backtest: ForecastBacktest
    levels: ExecutionLevels
    effects: EffectEstimator
    replay: LastYearCalendar
    scorer: CandidateScorer
    planners: Mapping[Planner, PromoPlanner]
    frontier: BudgetFrontier
    rules: PlanRules


class PromoService:
    """Facade: forecast -> options -> frontier -> plans -> horizon KPIs, per objective.

    Regular prices come from Phase 4's recommendations for the same objective; the
    plan is solved at the recommended budget (never above the available budget).
    """

    def __init__(
        self,
        parts: PromoParts,
        objectives: Sequence[Objective],
        budget: float,
        outputs: PromoOutputs,
    ) -> None:
        self._parts = parts
        self._objectives = list(objectives)
        self._budget = budget
        self._outputs = outputs

    def run(
        self,
        panel: pd.DataFrame,
        baseline: pd.DataFrame,
        final: pd.DataFrame,
        recommendations: pd.DataFrame,
    ) -> PromoReport:
        """Plan every objective present in the recommendations, then write all tables."""
        future = self._parts.calendar.build(panel)
        recs = recommendations.merge(final[[Col.UPC, Col.PROMO_ELASTICITY]], on=Col.UPC)
        first = recs[recs[Col.OBJECTIVE] == recs[Col.OBJECTIVE].iloc[0]]
        base_volume = first.set_index(Col.UPC)[kpi_col(Scenario.CURRENT, Metric.VOLUME)]
        backtest = self._parts.backtest.run(baseline)
        method = self._parts.backtest.choose(backtest)
        weekly = self._parts.forecaster.forecast(future, baseline, base_volume, method)
        levels = self._parts.levels.estimate(panel)
        logger.info("Tactic execution levels", extra={t.value: v for t, v in levels.items()})
        effects = self._parts.effects.estimate(future[future[Col.HORIZON_WEEK] == 1], levels)
        history = self._parts.replay.keys(panel, int(future[Col.HORIZON_WEEK].max()))
        blocks = [
            self._plan_objective(weekly, recs[recs[Col.OBJECTIVE] == o.value], effects, history, o)
            for o in self._objectives
            if (recs[Col.OBJECTIVE] == o.value).any()
        ]
        stacked = {
            name: pd.concat([b[name] for b in blocks], ignore_index=True)
            for name in ("forecast", "candidates", "plan", "frontier", "summary")
        }
        report = PromoReport(backtest=backtest, **stacked)
        self._write(report)
        return report

    def _plan_objective(
        self,
        weekly: pd.DataFrame,
        recs: pd.DataFrame,
        effects: PromoEffects,
        history: pd.DataFrame,
        objective: Objective,
    ) -> dict[str, pd.DataFrame]:
        """Forecast, options, frontier, plans and summary for one objective."""
        logger.info("Planning promotions", extra={"objective": objective.value})
        recs = recs.assign(**{Col.REGULAR_PRICE: recs[kpi_col(Scenario.RECOMMENDED, Metric.PRICE)]})
        indexed = recs.set_index(Col.UPC)
        ratio = (
            indexed[kpi_col(Scenario.RECOMMENDED, Metric.VOLUME)]
            / indexed[kpi_col(Scenario.CURRENT, Metric.VOLUME)]
        )
        forecast = weekly.assign(
            **{Col.FORECAST_BASELINE: weekly[Col.FORECAST_CURRENT] * weekly[Col.UPC].map(ratio)}
        )
        candidates = self._parts.scorer.score(forecast, recs, effects)
        frontier = self._parts.frontier.trace(candidates, self._budget)
        budget = float(frontier.loc[frontier[Col.RECOMMENDED_BUDGET], Col.BUDGET].iloc[0])
        plans = self._plans(candidates, history, budget)
        plan = pd.concat(
            [p.assign(**{Col.SCENARIO: s.value}) for s, p in plans.items()], ignore_index=True
        )
        tag = {Col.OBJECTIVE: objective.value}
        milp = plans[PromoScenario.MILP_PLAN]
        return {
            "forecast": self._calendar(forecast, recs, milp, effects).assign(**tag),
            "candidates": candidates.assign(**tag),
            "plan": plan.assign(**tag),
            "frontier": frontier.assign(**tag),
            "summary": self._summary(forecast, recs, plans).assign(**tag),
        }

    def _plans(
        self, candidates: pd.DataFrame, history: pd.DataFrame, budget: float
    ) -> dict[PromoScenario, pd.DataFrame]:
        """Last year's calendar, the MILP at its spend, and every planner at `budget`.

        Each plan carries the budget it was solved for; planner plans are rule-checked.
        """
        last_year = candidates[self._parts.replay.mask(candidates, history)]
        spend = float(last_year[Col.TRADE_SPEND].sum())
        milp = self._parts.planners[Planner.MILP]
        runs = {PromoScenario.MILP_SAME_SPEND: (milp, spend)} | {
            PLAN_SCENARIOS[name]: (planner, budget)
            for name, planner in self._parts.planners.items()
        }
        plans = {PromoScenario.LAST_YEAR_PLAN: last_year.assign(**{Col.BUDGET: spend})}
        for scenario, (planner, limit) in runs.items():
            chosen = candidates[planner.plan(candidates, limit)]
            broken = self._parts.rules.violations(chosen, limit)
            if broken:
                logger.warning("Plan breaks rules", extra={"plan": scenario.value, "rules": broken})
            plans[scenario] = chosen.assign(**{Col.BUDGET: limit})
        return plans

    @staticmethod
    def _calendar(
        forecast: pd.DataFrame, recs: pd.DataFrame, plan: pd.DataFrame, effects: PromoEffects
    ) -> pd.DataFrame:
        """Weekly product table: forecasts, the MILP plan and units with promo lifts and dips."""
        attrs = recs[[Col.UPC, Col.DESCRIPTION, Col.PRODUCT_SIZE, Col.CATEGORY,
                      Col.CURRENT_PRICE, Col.REGULAR_PRICE]]  # fmt: skip
        keys = [Col.UPC, Col.HORIZON_WEEK]
        table = (
            forecast.merge(attrs, on=Col.UPC)
            .merge(plan[[*keys, *PLAN_COLS]], on=keys, how="left")
            .sort_values(keys)
            .reset_index(drop=True)
        )
        units = table[Col.PROMO_UNITS].fillna(table[Col.FORECAST_BASELINE])
        promo = table[Col.TACTIC].notna().astype(float)
        for lag, col in enumerate(effects.pantry.columns, start=1):
            after = promo.groupby(table[Col.UPC]).shift(lag).fillna(0.0)
            dip = np.exp(table[Col.UPC].map(effects.pantry[col]).fillna(0.0))
            units = units * np.where(after > 0, dip, 1.0)
        return table.assign(**{Col.PLAN_UNITS: units})

    @staticmethod
    def _summary(
        forecast: pd.DataFrame, recs: pd.DataFrame, plans: Mapping[PromoScenario, pd.DataFrame]
    ) -> pd.DataFrame:
        """Horizon totals per scenario and % change against current prices with no promo."""
        prices = recs.set_index(Col.UPC)
        upc = forecast[Col.UPC]
        cost = upc.map(prices[Col.UNIT_COST])

        def no_promo(volume: pd.Series, price: pd.Series) -> dict[Col, float]:
            return {Col.VOLUME: volume.sum(), Col.REVENUE: (volume * price).sum(),
                    Col.PROFIT: (volume * (price - cost)).sum(), Col.TRADE_SPEND: 0.0,
                    Col.N_PROMOS: 0}  # fmt: skip

        current = no_promo(forecast[Col.FORECAST_CURRENT], upc.map(prices[Col.CURRENT_PRICE]))
        regular = no_promo(forecast[Col.FORECAST_BASELINE], upc.map(prices[Col.REGULAR_PRICE]))
        rows = {
            PromoScenario.CURRENT_NO_PROMO: current,
            PromoScenario.RECOMMENDED_NO_PROMO: regular,
        }
        for scenario, plan in plans.items():
            rows[scenario] = {
                Col.VOLUME: regular[Col.VOLUME] + plan[Col.INCREMENTAL].sum(),
                Col.REVENUE: regular[Col.REVENUE] + plan[Col.INCREMENTAL_REVENUE].sum(),
                Col.PROFIT: regular[Col.PROFIT] + plan[Col.INCREMENTAL_PROFIT].sum(),
                Col.TRADE_SPEND: plan[Col.TRADE_SPEND].sum(),
                Col.N_PROMOS: len(plan),
            }
        frame = pd.DataFrame.from_dict(rows, orient="index").rename_axis(Col.SCENARIO).reset_index()
        frame[Col.SCENARIO] = frame[Col.SCENARIO].map(lambda s: PromoScenario(s).value)
        for col in TOTALS:
            frame[f"{col.value}_change"] = frame[col] / current[col] - 1
        return frame

    def _write(self, report: PromoReport) -> None:
        """Persist every table as parquet."""
        for field in fields(PromoReport):
            path: Path = getattr(self._outputs, field.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            getattr(report, field.name).to_parquet(path, index=False)
        logger.info("Wrote promo plan", extra={"plan_rows": len(report.plan)})


class PromoFactory:
    """Builds a PromoService from config and the fitted elasticity models.

    The only place that instantiates forecast and promo classes.
    """

    def __init__(
        self, cfg: Config, models: Mapping[ModelName, ElasticityModel], outputs: PromoOutputs
    ) -> None:
        self._cfg = cfg
        self._models = models
        self._outputs = outputs

    def build(self) -> PromoService:
        """Wire calendar, forecaster, effects, scorer, planners and frontier."""
        cfg, promo = self._cfg, self._cfg.promo
        model = self._models[promo.lift_model]
        context = ForecastContext(
            cfg.forecast, cfg.optimization.reference_weeks, cfg.features.fourier_order,
            SeasonalIndex(model, cfg.features.fourier_order),
        )  # fmt: skip
        forecasters = {cls.name: cls(context) for cls in FORECASTERS}
        planners = {cls.name: cls.from_config(promo) for cls in PLANNERS}
        parts = PromoParts(
            calendar=FutureCalendar(
                cfg.forecast.horizon_weeks,
                [FourierSeasonality.from_config(cfg), HolidayFlags.from_config(cfg)],
                cfg.features.pantry_lags,
            ),
            forecaster=BaselineForecaster(forecasters),
            backtest=ForecastBacktest(forecasters, cfg.forecast.method),
            levels=ExecutionLevels(promo.tactics, cfg.features.promo_min_tactic_share),
            effects=EffectEstimator(model, cfg.features.pantry_lags),
            replay=LastYearCalendar(cfg.features.promo_min_tactic_share, promo.depths),
            scorer=CandidateScorer(promo.tactics, promo.depths, promo.fixed_cost),
            planners=planners,
            frontier=BudgetFrontier(
                planners[Planner.MILP], promo.frontier_points, promo.min_marginal_return
            ),
            rules=PlanRules.from_config(promo),
        )
        return PromoService(parts, cfg.optimization.objectives, promo.budget, self._outputs)
