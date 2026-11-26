"""Feature engineering steps and the time-based split for the panel.

Each FeatureStep registers in FEATURE_STEPS and FeatureBuilder applies them in
registration order: logs and price split, coverage, promo flag, pantry loading,
Fourier seasonality, US holiday weeks and a global time index with the split.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import holidays
import numpy as np
import pandas as pd

from src.config import Config
from src.schema import TACTIC_COLS, Col, Split, fourier_columns, pantry_columns
from src.utils import ConfigError, Registry

WEEK_DAYS = 7
DAYS_PER_YEAR = 365.25


class FeatureStep(ABC):
    """One feature transformation over the panel; returns a new DataFrame.

    Subclasses declare a unique `name` and register in FEATURE_STEPS.
    """

    name: ClassVar[str]

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: Config) -> FeatureStep:
        """Build the step from config."""

    @abstractmethod
    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return panel with this step's columns added."""


FEATURE_STEPS: Registry[str, type[FeatureStep]] = Registry("name")


@FEATURE_STEPS.register
class LogFeatures(FeatureStep):
    """Adds ln_units, ln_price, ln_base_price, ln_promo_price and discount_depth.

    ln_price = ln_base_price + ln_promo_price: regular price level plus promo deviation.
    """

    name = "log"

    @classmethod
    def from_config(cls, cfg: Config) -> LogFeatures:
        """No settings needed."""
        return cls()

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Add the log and discount columns."""
        return panel.assign(
            **{
                Col.LN_UNITS: np.log(panel[Col.UNITS]),
                Col.LN_PRICE: np.log(panel[Col.PRICE]),
                Col.LN_BASE_PRICE: np.log(panel[Col.BASE_PRICE]),
                Col.LN_PROMO_PRICE: np.log(panel[Col.PRICE] / panel[Col.BASE_PRICE]),
                Col.DISCOUNT_DEPTH: (1 - panel[Col.PRICE] / panel[Col.BASE_PRICE]).clip(0, 1),
            }
        )


@FEATURE_STEPS.register
class StoreCoverage(FeatureStep):
    """Adds ln_n_stores: distribution control, since summed units scale with coverage.

    Constant (0) at store grain, where models drop it as zero-variance.
    """

    name = "store_coverage"

    @classmethod
    def from_config(cls, cfg: Config) -> StoreCoverage:
        """No settings needed."""
        return cls()

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Add the log store count."""
        return panel.assign(**{Col.LN_N_STORES: np.log(panel[Col.N_STORES])})


@FEATURE_STEPS.register
class PromoFlag(FeatureStep):
    """Flags promo weeks: discount depth or feature/display share above thresholds.

    Mirrors the RGMx promo flag (price/EDLP below threshold or promo ACV above one).
    """

    name = "promo_flag"

    def __init__(self, min_discount: float, min_tactic_share: float) -> None:
        self._min_discount = min_discount
        self._min_share = min_tactic_share

    @classmethod
    def from_config(cls, cfg: Config) -> PromoFlag:
        """Read thresholds from config."""
        return cls(cfg.features.promo_min_discount, cfg.features.promo_min_tactic_share)

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Add promo_flag (0/1)."""
        merch = [c for c in TACTIC_COLS if c != Col.TPR]
        is_promo = (panel[Col.DISCOUNT_DEPTH] >= self._min_discount) | (
            panel[merch].max(axis=1) >= self._min_share
        )
        return panel.assign(**{Col.PROMO_FLAG: is_promo.astype(int)})


@FEATURE_STEPS.register
class PantryLoading(FeatureStep):
    """Adds pl1..plN: the promo flag lagged 1..N weeks within each series.

    Captures the post-promotion dip as shoppers use up stock bought on deal (RGMx).
    """

    name = "pantry_loading"
    SERIES_COLS: ClassVar[tuple[Col, ...]] = (Col.STORE, Col.UPC)

    def __init__(self, lags: int) -> None:
        self._lags = lags

    @classmethod
    def from_config(cls, cfg: Config) -> PantryLoading:
        """Read the number of lags from config."""
        return cls(cfg.features.pantry_lags)

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Add lagged promo flags (0 before the series starts); keeps row order."""
        keys = [c for c in self.SERIES_COLS if c in panel.columns]
        ordered = panel.sort_values([*keys, Col.WEEK])
        flags = ordered.groupby(keys)[Col.PROMO_FLAG]
        lags = {
            name: flags.shift(i).fillna(0).astype(int)
            for i, name in enumerate(pantry_columns(self._lags), start=1)
        }
        return panel.assign(**{name: lag.reindex(panel.index) for name, lag in lags.items()})


@FEATURE_STEPS.register
class FourierSeasonality(FeatureStep):
    """Yearly seasonality as sin/cos pairs of day-of-year.

    `column_names()` lists the generated columns for model formulas.
    """

    name = "fourier"

    def __init__(self, order: int) -> None:
        self._order = order

    @classmethod
    def from_config(cls, cfg: Config) -> FourierSeasonality:
        """Read the Fourier order from config."""
        return cls(cfg.features.fourier_order)

    def column_names(self) -> list[str]:
        """sin_1, cos_1, ..., sin_K, cos_K."""
        return fourier_columns(self._order)

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Add 2 * order seasonality columns."""
        angle = 2 * np.pi * panel[Col.WEEK].dt.day_of_year.to_numpy() / DAYS_PER_YEAR
        names = iter(self.column_names())
        terms = {}
        for k in range(1, self._order + 1):
            terms[next(names)] = np.sin(k * angle)
            terms[next(names)] = np.cos(k * angle)
        return panel.assign(**terms)


@FEATURE_STEPS.register
class HolidayFlags(FeatureStep):
    """Flags weeks whose 7 days (ending on the week date) contain a holiday.

    holiday_week = any public holiday; named flags match holiday names.
    """

    name = "holidays"
    NAMED: ClassVar[dict[Col, str]] = {
        Col.THANKSGIVING: "Thanksgiving",
        Col.CHRISTMAS: "Christmas Day",
        Col.SUPER_BOWL: "Super Bowl",
    }
    _DATE, _NAME, _PUBLIC = "date", "name", "is_public"

    def __init__(self, country: str) -> None:
        self._country = country

    @classmethod
    def from_config(cls, cfg: Config) -> HolidayFlags:
        """Read the holiday country from config."""
        return cls(cfg.features.holiday_country)

    @staticmethod
    def super_bowl_date(year: int) -> dt.date:
        """First Sunday of February (Super Bowl date through the 2021 season)."""
        feb1 = dt.date(year, 2, 1)
        return feb1 + dt.timedelta(days=(6 - feb1.weekday()) % 7)

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Add holiday_week and named-holiday flags (0/1) by week."""
        weeks = pd.DataFrame({Col.WEEK: np.sort(panel[Col.WEEK].unique())})
        weeks[Col.WEEK] = weeks[Col.WEEK].astype("datetime64[ns]")
        events = self._events(range(weeks[Col.WEEK].min().year - 1, weeks[Col.WEEK].max().year + 1))
        matched = pd.merge_asof(
            events, weeks, left_on=self._DATE, right_on=Col.WEEK,
            direction="forward", tolerance=pd.Timedelta(days=WEEK_DAYS - 1),
        ).dropna(subset=[Col.WEEK])  # fmt: skip
        flags = matched.assign(
            **{Col.HOLIDAY_WEEK: matched[self._PUBLIC]},
            **{
                col: matched[self._NAME].str.contains(n, regex=False)
                for col, n in self.NAMED.items()
            },
        )
        flag_cols = [Col.HOLIDAY_WEEK, *self.NAMED]
        by_week = flags.groupby(Col.WEEK)[flag_cols].max()
        by_week = weeks.merge(by_week, on=Col.WEEK, how="left").fillna(False)
        by_week[flag_cols] = by_week[flag_cols].astype(int)
        by_week[Col.WEEK] = by_week[Col.WEEK].astype(panel[Col.WEEK].dtype)
        return panel.merge(by_week, on=Col.WEEK, how="left")

    def _events(self, years: range) -> pd.DataFrame:
        """Public holidays plus Super Bowl Sundays, sorted by date."""
        cal = holidays.country_holidays(self._country, years=years)
        public = pd.DataFrame({self._DATE: list(cal.keys()), self._NAME: list(cal.values())})
        bowl = pd.DataFrame({self._DATE: [self.super_bowl_date(y) for y in years]})
        bowl[self._NAME] = self.NAMED[Col.SUPER_BOWL]
        events = pd.concat(
            [public.assign(**{self._PUBLIC: True}), bowl.assign(**{self._PUBLIC: False})]
        )
        events[self._DATE] = pd.to_datetime(events[self._DATE]).astype("datetime64[ns]")
        return events.sort_values(self._DATE).reset_index(drop=True)


@FEATURE_STEPS.register
class TimeSplit(FeatureStep):
    """Adds a global week index `t` and `split` (last N distinct weeks = test).

    A global cutoff (not per product) keeps every test week after every train week.
    """

    name = "time_split"

    def __init__(self, holdout_weeks: int) -> None:
        self._holdout_weeks = holdout_weeks

    @classmethod
    def from_config(cls, cfg: Config) -> TimeSplit:
        """Read holdout length from config."""
        return cls(cfg.split.holdout_weeks)

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Add t and split; raises ConfigError if the holdout leaves no training weeks."""
        weeks = np.sort(panel[Col.WEEK].unique())
        if self._holdout_weeks >= len(weeks):
            raise ConfigError(f"holdout_weeks={self._holdout_weeks} leaves no training weeks")
        cutoff = weeks[-self._holdout_weeks]
        t = ((panel[Col.WEEK] - weeks[0]).dt.days // WEEK_DAYS).astype(int)
        split = np.where(panel[Col.WEEK] >= cutoff, Split.TEST.value, Split.TRAIN.value)
        return panel.assign(**{Col.T: t, Col.SPLIT: split})


class FeatureBuilder:
    """Applies a sequence of FeatureSteps in order.

    Trend is not a step here because it depends on the split (see LowessTrend).
    """

    def __init__(self, steps: Sequence[FeatureStep]) -> None:
        self._steps = list(steps)

    def build(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return panel with every step applied."""
        for step in self._steps:
            panel = step.apply(panel)
        return panel
