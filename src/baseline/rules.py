"""RGMx bad-baseline rules and the corrector that applies them.

Each rule flags weeks where the raw top-down baseline is implausible and states
its fix: replace with actual volume, or with a centred rolling mean of the raw
baseline. Rules register in BASELINE_RULES; rolling fixes take precedence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pandas as pd

from src.config import BaselineConfig
from src.schema import HOLIDAY_COLS, BaselineFix, Col
from src.utils import AppLogger, Registry

logger = AppLogger.get_logger(__name__)

PREV, NEXT, RECENT_PROMO, HOLIDAY = "base_prev", "base_next", "promo_recent", "holiday_any"
REL_TOL = 1e-9  # exp(ln(units)) can exceed units by float noise


class BaselineRule(ABC):
    """One RGMx check on the raw baseline; flagged weeks get the rule's fix.

    Rules read a prepared view with neighbour baselines and recent-promo/holiday flags.
    """

    name: ClassVar[str]
    fix: ClassVar[BaselineFix]

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: BaselineConfig) -> BaselineRule:
        """Build the rule from config."""

    @abstractmethod
    def flags(self, view: pd.DataFrame) -> pd.Series:
        """Boolean Series: True where the week breaks the rule."""

    @property
    def column(self) -> str:
        """Output column holding this rule's flag."""
        return f"rule_{self.name}"


BASELINE_RULES: Registry[str, type[BaselineRule]] = Registry("name")


@BASELINE_RULES.register
class ZeroBaseline(BaselineRule):
    """Rule 2: baseline is zero while the product sold (fix: actual volume)."""

    name = "zero_baseline"
    fix = BaselineFix.ACTUAL

    @classmethod
    def from_config(cls, cfg: BaselineConfig) -> ZeroBaseline:
        """No settings."""
        return cls()

    def flags(self, view: pd.DataFrame) -> pd.Series:
        """base == 0 and units > 0."""
        return (view[Col.BASE_RAW] <= 0) & (view[Col.UNITS] > 0)


@BASELINE_RULES.register
class ImplausiblyHigh(BaselineRule):
    """Rule 3: baseline far above actual, or above it with no recent promo (fix: actual)."""

    name = "implausibly_high"
    fix = BaselineFix.ACTUAL

    def __init__(self, max_ratio: float) -> None:
        self._max_ratio = max_ratio

    @classmethod
    def from_config(cls, cfg: BaselineConfig) -> ImplausiblyHigh:
        """Read the maximum baseline-to-volume ratio."""
        return cls(cfg.max_base_to_volume)

    def flags(self, view: pd.DataFrame) -> pd.Series:
        """base/units > max ratio, or base > units with no promo in the lookback."""
        too_high = view[Col.BASE_RAW] / view[Col.UNITS] > self._max_ratio
        above = view[Col.BASE_RAW] > view[Col.UNITS] * (1 + REL_TOL)
        above_without_promo = above & ~view[RECENT_PROMO]
        return too_high | above_without_promo


@BASELINE_RULES.register
class PromoDip(BaselineRule):
    """Rule 4: promo-week baseline below both neighbours (fix: rolling mean)."""

    name = "promo_dip"
    fix = BaselineFix.ROLLING

    @classmethod
    def from_config(cls, cfg: BaselineConfig) -> PromoDip:
        """No settings."""
        return cls()

    def flags(self, view: pd.DataFrame) -> pd.Series:
        """Promo week and base < previous and < next week's base."""
        base = view[Col.BASE_RAW]
        return (view[Col.PROMO_FLAG] == 1) & (base < view[PREV]) & (base < view[NEXT])


@BASELINE_RULES.register
class PromoSpike(BaselineRule):
    """Rule 5: non-holiday promo-week baseline above both neighbours (fix: rolling mean)."""

    name = "promo_spike"
    fix = BaselineFix.ROLLING

    @classmethod
    def from_config(cls, cfg: BaselineConfig) -> PromoSpike:
        """No settings."""
        return cls()

    def flags(self, view: pd.DataFrame) -> pd.Series:
        """Promo week, no holiday, base > previous and > next week's base."""
        base = view[Col.BASE_RAW]
        spike = (base > view[PREV]) & (base > view[NEXT])
        return (view[Col.PROMO_FLAG] == 1) & ~view[HOLIDAY] & spike


@BASELINE_RULES.register
class PromoDrop(BaselineRule):
    """Rule 6: promo-week baseline well below a neighbour (fix: rolling mean)."""

    name = "promo_drop"
    fix = BaselineFix.ROLLING

    def __init__(self, ratio: float) -> None:
        self._ratio = ratio

    @classmethod
    def from_config(cls, cfg: BaselineConfig) -> PromoDrop:
        """Read the neighbour drop ratio."""
        return cls(cfg.neighbor_drop_ratio)

    def flags(self, view: pd.DataFrame) -> pd.Series:
        """Promo week and base < ratio x previous or next week's base."""
        base = view[Col.BASE_RAW]
        drop = (base < self._ratio * view[PREV]) | (base < self._ratio * view[NEXT])
        return (view[Col.PROMO_FLAG] == 1) & drop


class BaselineCorrector:
    """Applies the rules and writes the final baseline and how it was fixed.

    Rolling-mean fixes win over actual-volume fixes (RGMx precedence). Series edges
    have no neighbour, so neighbour rules never fire there.
    """

    SERIES_COLS: ClassVar[tuple[Col, ...]] = (Col.STORE, Col.UPC)

    def __init__(self, rules: Sequence[BaselineRule], rolling_window: int, lookback: int) -> None:
        self._rules = list(rules)
        self._window = rolling_window
        self._lookback = lookback

    def correct(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Frame plus one flag column per rule, baseline_units and baseline_fix."""
        keys = [c for c in self.SERIES_COLS if c in frame.columns]
        ordered = frame.sort_values([*keys, Col.WEEK])
        view = self._view(ordered, keys)
        flags = {rule.column: rule.flags(view).fillna(False) for rule in self._rules}
        rolling_hit = self._any(flags, BaselineFix.ROLLING, ordered.index)
        actual_hit = self._any(flags, BaselineFix.ACTUAL, ordered.index) & ~rolling_hit
        rolling = ordered.groupby(keys)[Col.BASE_RAW].transform(
            lambda s: s.rolling(self._window, center=True, min_periods=1).mean()
        )
        baseline = np.select(
            [rolling_hit, actual_hit], [rolling, ordered[Col.UNITS]], default=ordered[Col.BASE_RAW]
        )
        fix = np.select(
            [rolling_hit, actual_hit],
            [BaselineFix.ROLLING.value, BaselineFix.ACTUAL.value],
            default=BaselineFix.NONE.value,
        )
        corrected = ordered.assign(**flags, **{Col.BASELINE: baseline, Col.BASELINE_FIX: fix})
        counts = pd.Series(fix).value_counts().to_dict()
        logger.info("Corrected baselines", extra={"weeks_by_fix": counts})
        return corrected.loc[frame.index]

    def _view(self, ordered: pd.DataFrame, keys: list[Col]) -> pd.DataFrame:
        """Neighbour baselines, recent-promo and holiday flags for the rules."""
        series = ordered.groupby(keys)
        promo = series[Col.PROMO_FLAG]
        recent = sum(promo.shift(i).fillna(0) for i in range(1, self._lookback + 1)) > 0
        holiday_cols = [c for c in HOLIDAY_COLS if c in ordered.columns]
        return ordered.assign(
            **{
                PREV: series[Col.BASE_RAW].shift(1),
                NEXT: series[Col.BASE_RAW].shift(-1),
                RECENT_PROMO: recent,
                HOLIDAY: ordered[holiday_cols].sum(axis=1) > 0,
            }
        )

    def _any(self, flags: dict[str, pd.Series], fix: BaselineFix, index: pd.Index) -> pd.Series:
        """True where any rule with the given fix flagged the week."""
        hits = [flags[r.column] for r in self._rules if r.fix is fix]
        if not hits:
            return pd.Series(False, index=index)
        return pd.concat(hits, axis=1).any(axis=1)
