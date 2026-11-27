"""Promo calendar planners: choose which options to run under a trade budget.

PlanRules holds the business rules (promos per SKU, gap, feature and display slots)
and checks any plan. MilpPlanner solves the exact MILP with PuLP + HiGHS;
GreedyPlanner is the ROI-ranking benchmark; LastYearCalendar replays history.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
import pulp

from src.config import PromoConfig
from src.schema import TACTIC_TERMS, Col, Planner, Tactic
from src.utils import AppLogger, Registry, SolverError

logger = AppLogger.get_logger(__name__)

MIN_GAIN = 1e-6  # options with no positive gain are never worth a slot
SOLVED: tuple[pulp.LpSolveStatus, ...] = (
    pulp.LpSolveStatus.Optimal, pulp.LpSolveStatus.GapLimit, pulp.LpSolveStatus.TimeLimit,
)  # fmt: skip


def _uses(column: Col) -> set[str]:
    """Tactic values that occupy a slot of the given merchandising type."""
    return {t.value for t, cols in TACTIC_TERMS.items() if column in cols}


FEATURE_TACTICS, DISPLAY_TACTICS = _uses(Col.FEATURE), _uses(Col.DISPLAY)


@dataclass(frozen=True)
class PlanRules:
    """Calendar rules every plan must respect (besides the budget).

    At most one promo per SKU in any (gap + 1)-week window, at most `max_promos`
    per SKU, and weekly caps on feature (circular) and display slots.
    """

    max_promos: int
    min_gap_weeks: int
    feature_slots: int
    display_slots: int

    @classmethod
    def from_config(cls, cfg: PromoConfig) -> PlanRules:
        """Read the rules from promo config."""
        return cls(cfg.max_promos_per_sku, cfg.min_gap_weeks, cfg.max_feature_slots,
                   cfg.max_display_slots)  # fmt: skip

    @property
    def window(self) -> int:
        """Length of the rolling window allowed to hold one promo per SKU."""
        return self.min_gap_weeks + 1

    def violations(self, plan: pd.DataFrame, budget: float) -> list[str]:
        """Human-readable rule breaches of a chosen plan (empty when valid)."""
        found = []
        if plan[Col.TRADE_SPEND].sum() > budget * (1 + 1e-9) + 1e-6:
            found.append("budget")
        if (plan.groupby(Col.UPC).size() > self.max_promos).any():
            found.append("max_promos")
        for _, weeks in plan.groupby(Col.UPC)[Col.HORIZON_WEEK]:
            if (np.diff(np.sort(weeks.to_numpy())) < self.window).any():
                found.append("min_gap")
                break
        slot_caps = (
            (FEATURE_TACTICS, self.feature_slots, "feature_slots"),
            (DISPLAY_TACTICS, self.display_slots, "display_slots"),
        )
        for tactics, cap, name in slot_caps:
            used = plan[plan[Col.TACTIC].isin(tactics)].groupby(Col.HORIZON_WEEK).size()
            if (used > cap).any():
                found.append(name)
        return found


class PromoPlanner(ABC):
    """Chooses a subset of promo options maximising total gain within the budget.

    `plan` returns a boolean mask aligned to the candidates' rows.
    """

    name: ClassVar[Planner]

    def __init__(self, rules: PlanRules) -> None:
        self._rules = rules

    @classmethod
    def from_config(cls, cfg: PromoConfig) -> PromoPlanner:
        """Build with the configured calendar rules."""
        return cls(PlanRules.from_config(cfg))

    @abstractmethod
    def plan(self, candidates: pd.DataFrame, budget: float) -> np.ndarray:
        """Boolean mask of chosen options (one objective, one set of prices)."""


PLANNERS: Registry[Planner, type[PromoPlanner]] = Registry("name")


@PLANNERS.register
class MilpPlanner(PromoPlanner):
    """Exact plan: binary x per option, max sum(gain x) s.t. budget and calendar rules.

    Only options with positive gain become variables (others are never optimal).
    Solved with HiGHS through PuLP; stops at the relative MIP gap or time limit.
    """

    name = Planner.MILP

    def __init__(self, rules: PlanRules, mip_gap: float = 1e-3, time_limit: float = 60) -> None:
        super().__init__(rules)
        self._mip_gap = mip_gap
        self._time_limit = time_limit

    @classmethod
    def from_config(cls, cfg: PromoConfig) -> MilpPlanner:
        """Rules plus the MIP gap and time limit."""
        return cls(PlanRules.from_config(cfg), cfg.mip_gap, cfg.time_limit_seconds)

    def plan(self, candidates: pd.DataFrame, budget: float) -> np.ndarray:
        """Solve the MILP; raises SolverError if HiGHS returns no solution."""
        pool = np.flatnonzero(candidates[Col.GAIN].to_numpy() > MIN_GAIN)
        mask = np.zeros(len(candidates), dtype=bool)
        if pool.size == 0:
            return mask
        rows = candidates.iloc[pool].reset_index(drop=True)
        model = pulp.LpProblem("promo_plan", pulp.LpMaximize)
        x = [model.add_variable(f"x_{i}", cat="Binary") for i in range(len(rows))]
        model += pulp.lpSum(g * v for g, v in zip(rows[Col.GAIN], x, strict=True))
        self._add_rules(model, rows, x, budget)
        stats = model.solve(pulp.HiGHS(msg=False, gapRel=self._mip_gap, timeLimit=self._time_limit))
        if stats.status not in SOLVED or not stats.has_solution:
            raise SolverError(f"Promo MILP returned {stats.status.name}")
        chosen = np.array([(v.value() or 0) > 0.5 for v in x])
        mask[pool[chosen]] = True
        return mask

    def _add_rules(self, model: pulp.LpProblem, rows: pd.DataFrame, x: list, budget: float) -> None:
        """Budget (skipped when infinite), promos per SKU, one promo per window, slot caps."""
        rules = self._rules
        if math.isfinite(budget):
            spend = zip(rows[Col.TRADE_SPEND], x, strict=True)
            model += pulp.lpSum(s * v for s, v in spend) <= budget
        by_week: dict[tuple, list] = defaultdict(list)
        for i, (upc, week) in enumerate(zip(rows[Col.UPC], rows[Col.HORIZON_WEEK], strict=True)):
            by_week[(upc, week)].append(x[i])
        for upc, idx in rows.groupby(Col.UPC).indices.items():
            model += pulp.lpSum(x[i] for i in idx) <= rules.max_promos
            weeks = sorted({w for (u, w) in by_week if u == upc})
            for start in weeks:
                window = [v for w in range(start, start + rules.window)
                          for v in by_week.get((upc, w), [])]  # fmt: skip
                model += pulp.lpSum(window) <= 1
        for tactics, cap in ((FEATURE_TACTICS, rules.feature_slots),
                             (DISPLAY_TACTICS, rules.display_slots)):  # fmt: skip
            slot_rows = rows[rows[Col.TACTIC].isin(tactics)]
            for _, idx in slot_rows.groupby(Col.HORIZON_WEEK).indices.items():
                model += pulp.lpSum(x[i] for i in slot_rows.index[idx]) <= cap


@PLANNERS.register
class GreedyPlanner(PromoPlanner):
    """Benchmark: take options by highest gain per dollar while every rule still holds.

    The usual spreadsheet heuristic; it can strand budget and miss better combinations.
    """

    name = Planner.GREEDY

    def plan(self, candidates: pd.DataFrame, budget: float) -> np.ndarray:
        """Greedy pass in descending ROI (free options first)."""
        rules = self._rules
        ratio = candidates[Col.GAIN] / candidates[Col.TRADE_SPEND].where(
            candidates[Col.TRADE_SPEND] > 0
        )
        order = ratio.fillna(math.inf).where(candidates[Col.GAIN] > MIN_GAIN).dropna()
        weeks: dict[object, list[int]] = defaultdict(list)
        slots: dict[tuple[str, int], int] = defaultdict(int)
        caps = {"feature": rules.feature_slots, "display": rules.display_slots}
        left, mask = budget, np.zeros(len(candidates), dtype=bool)
        for pos in candidates.index.get_indexer(order.sort_values(ascending=False).index):
            row = candidates.iloc[pos]
            upc, week, spend = row[Col.UPC], int(row[Col.HORIZON_WEEK]), row[Col.TRADE_SPEND]
            kinds = self._slot_kinds(row[Col.TACTIC])
            if (
                spend > left
                or len(weeks[upc]) >= rules.max_promos
                or any(abs(week - w) < rules.window for w in weeks[upc])
                or any(slots[(k, week)] >= caps[k] for k in kinds)
            ):
                continue
            mask[pos], left = True, left - spend
            weeks[upc].append(week)
            for k in kinds:
                slots[(k, week)] += 1
        return mask

    @staticmethod
    def _slot_kinds(tactic: str) -> list[str]:
        """Merchandising slots ('feature', 'display') the tactic occupies."""
        return [k for k, group in (("feature", FEATURE_TACTICS), ("display", DISPLAY_TACTICS))
                if tactic in group]  # fmt: skip


class LastYearCalendar:
    """Business-as-usual benchmark: the same weeks one year earlier, mapped to options.

    Each historical promo week becomes the option with its tactic (merchandising
    shares >= threshold; otherwise TPR) and the nearest configured depth.
    """

    YEAR: ClassVar[pd.Timedelta] = pd.Timedelta(weeks=52)

    def __init__(self, min_share: float, depths: list[float]) -> None:
        self._min_share = min_share
        self._depths = np.asarray(depths)

    def keys(self, panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """UPC, horizon week, tactic and depth of last year's promos for the horizon weeks."""
        end = panel[Col.WEEK].max()
        step = ((panel[Col.WEEK] - (end - self.YEAR)) / pd.Timedelta(weeks=1)).round()
        promo = panel[(panel[Col.PROMO_FLAG] == 1) & step.between(1, horizon)]
        feature = promo[Col.FEATURE] >= self._min_share
        display = promo[Col.DISPLAY] >= self._min_share
        tactic = np.select(
            [feature & display, feature, display],
            [Tactic.FEATURE_DISPLAY.value, Tactic.FEATURE.value, Tactic.DISPLAY.value],
            default=Tactic.TPR.value,
        )
        nearest = np.abs(promo[Col.DISCOUNT_DEPTH].to_numpy()[:, None] - self._depths).argmin(1)
        return pd.DataFrame({
            Col.UPC: promo[Col.UPC].to_numpy(), Col.HORIZON_WEEK: step[promo.index].astype(int),
            Col.TACTIC: tactic, Col.DISCOUNT_DEPTH: self._depths[nearest],
        })  # fmt: skip

    @staticmethod
    def mask(candidates: pd.DataFrame, keys: pd.DataFrame) -> np.ndarray:
        """Boolean mask of the candidate rows matching last year's promos."""
        cols = [Col.UPC, Col.HORIZON_WEEK, Col.TACTIC, Col.DISCOUNT_DEPTH]
        hit = candidates[cols].merge(keys.drop_duplicates(), on=cols, how="left", indicator=True)
        return (hit["_merge"] == "both").to_numpy()
