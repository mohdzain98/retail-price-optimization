"""Live promo planning: re-plan the MILP with the user's budget, rules, tactics and fees.

PromoSimulator starts from the stored promo options of one objective, re-prices
them for changed fixed fees, keeps the allowed tactics and depths, then plans with
the MILP and compares against last year's calendar and no promotion.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config
from src.promo.frontier import BudgetFrontier
from src.promo.kpis import HorizonKpis
from src.promo.planner import LastYearCalendar, MilpPlanner, PlanRules
from src.schema import Col, PromoScenario
from src.simulator.data import SimulatorData

YOUR_PLAN = "your_plan"
KEYS: tuple[Col, ...] = (Col.UPC, Col.HORIZON_WEEK, Col.TACTIC, Col.DISCOUNT_DEPTH)
NAMES: tuple[Col, ...] = (Col.UPC, Col.DESCRIPTION, Col.PRODUCT_SIZE, Col.CATEGORY)


@dataclass(frozen=True)
class PromoSettings:
    """User budget, calendar rules, allowed options and fees (hashable for caching).

    `fees` is a tuple of (tactic, fixed fee per SKU-week) pairs.
    """

    budget: float
    max_promos: int
    min_gap_weeks: int
    feature_slots: int
    display_slots: int
    tactics: tuple[str, ...]
    depths: tuple[float, ...]
    fees: tuple[tuple[str, float], ...]

    @classmethod
    def from_config(cls, cfg: Config) -> PromoSettings:
        """The pipeline's own promo settings."""
        promo = cfg.promo
        return cls(
            promo.budget,
            promo.max_promos_per_sku,
            promo.min_gap_weeks,
            promo.max_feature_slots,
            promo.max_display_slots,
            tuple(t.value for t in promo.tactics),
            tuple(promo.depths),
            tuple((t.value, fee) for t, fee in promo.fixed_cost.items()),
        )

    @property
    def rules(self) -> PlanRules:
        """Calendar rules for the planner."""
        return PlanRules(
            self.max_promos, self.min_gap_weeks, self.feature_slots, self.display_slots
        )


@dataclass(frozen=True)
class PromoResult:
    """A planned calendar with its scenario KPIs and weekly view.

    `summary` compares current and recommended prices without promotion, last
    year's calendar and the user's plan; `weekly` is one row per product-week.
    """

    plan: pd.DataFrame
    summary: pd.DataFrame
    weekly: pd.DataFrame


class PromoSimulator:
    """Re-plans promotions from the stored options of each objective.

    Options keep their stored lift and volumes; only fees, allowed options, budget
    and rules change, so re-planning takes about a second.
    """

    def __init__(self, data: SimulatorData) -> None:
        self._data = data
        self._kpis = HorizonKpis()

    def candidates(self, objective: str, settings: PromoSettings) -> pd.DataFrame:
        """Allowed options of `objective` with spend, profit and gain at the user's fees."""
        frame = self._data.candidates
        frame = frame[
            (frame[Col.OBJECTIVE] == objective)
            & frame[Col.TACTIC].isin(settings.tactics)
            & np.isclose(frame[Col.DISCOUNT_DEPTH].to_numpy()[:, None], settings.depths).any(1)
        ].reset_index(drop=True)
        return self._refee(frame, dict(settings.fees))

    def plan(self, objective: str, settings: PromoSettings) -> PromoResult:
        """MILP plan at the user's budget, compared with last year and no promotion."""
        options = self.candidates(objective, settings)
        chosen = options[MilpPlanner(settings.rules).plan(options, settings.budget)]
        history = self._data.plans
        history = history[
            (history[Col.OBJECTIVE] == objective)
            & (history[Col.SCENARIO] == PromoScenario.LAST_YEAR_PLAN.value)
        ]
        everything = self._refee(
            self._data.candidates[self._data.candidates[Col.OBJECTIVE] == objective],
            dict(settings.fees),
        )
        last_year = everything[LastYearCalendar.mask(everything, history[list(KEYS)])]
        forecast = self._forecast(objective)
        summary = self._kpis.summary(
            forecast,
            self._prices(objective),
            {PromoScenario.LAST_YEAR_PLAN.value: last_year, YOUR_PLAN: chosen},
        )
        return PromoResult(chosen, summary, self._weekly(forecast, chosen))

    def frontier(
        self, objective: str, settings: PromoSettings, points: int, hurdle: float
    ) -> pd.DataFrame:
        """Budget frontier for the user's options and rules, up to the unconstrained spend."""
        frontier = BudgetFrontier(MilpPlanner(settings.rules), points, hurdle)
        return frontier.trace(self.candidates(objective, settings), settings.budget)

    @staticmethod
    def _refee(frame: pd.DataFrame, fees: Mapping[str, float]) -> pd.DataFrame:
        """Re-price options for new fixed fees (gain charges fees in proportion c_eff / c)."""
        new_fee = frame[Col.TACTIC].map(fees).fillna(frame[Col.FIXED_COST])
        delta = new_fee - frame[Col.FIXED_COST]
        weight = frame[Col.EFFECTIVE_COST] / frame[Col.UNIT_COST]
        spend = frame[Col.TRADE_SPEND] + delta
        gain = frame[Col.GAIN] - weight * delta
        return frame.assign(
            **{
                Col.FIXED_COST: new_fee,
                Col.TRADE_SPEND: spend,
                Col.INCREMENTAL_PROFIT: frame[Col.INCREMENTAL_PROFIT] - delta,
                Col.GAIN: gain,
                Col.ROI: gain / spend.where(spend > 0),
            }
        )

    def _forecast(self, objective: str) -> pd.DataFrame:
        """Stored weekly forecast rows of `objective`."""
        forecast = self._data.forecast
        return forecast[forecast[Col.OBJECTIVE] == objective].reset_index(drop=True)

    def _prices(self, objective: str) -> pd.DataFrame:
        """UPC, current and regular price (from the forecast) and unit cost (Phase 4)."""
        recs = self._data.recommendations
        cost = recs.loc[recs[Col.OBJECTIVE] == objective, [Col.UPC, Col.UNIT_COST]]
        prices = self._forecast(objective)[[Col.UPC, Col.CURRENT_PRICE, Col.REGULAR_PRICE]]
        return prices.drop_duplicates(Col.UPC).merge(cost, on=Col.UPC)

    @staticmethod
    def _weekly(forecast: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
        """Product-week rows: baseline at recommended prices and units with the plan.

        A promo week shows its promo units; the pantry dip is netted in that same week.
        """
        cols = [*NAMES, Col.WEEK, Col.HORIZON_WEEK, Col.FORECAST_CURRENT, Col.FORECAST_BASELINE]
        keys = [Col.UPC, Col.HORIZON_WEEK]
        picked = plan[[*keys, Col.TACTIC, Col.DISCOUNT_DEPTH, Col.INCREMENTAL]]
        weekly = forecast[cols].merge(picked, on=keys, how="left")
        units = weekly[Col.FORECAST_BASELINE] + weekly[Col.INCREMENTAL].fillna(0.0)
        return weekly.assign(**{Col.PLAN_UNITS: units}).drop(columns=Col.INCREMENTAL)
