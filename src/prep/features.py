"""Feature engineering steps and the time-based split for the panel.

Each FeatureStep registers in FEATURE_STEPS and FeatureBuilder applies them in
registration order: log quantities/prices, Fourier seasonality, US holiday week
flags (incl. Super Bowl) and a global time index with the train/test split.
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
from src.schema import Col, Split
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
    """Adds ln_units, ln_price, ln_base_price and discount_depth (clipped to [0, 1])."""

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
                Col.DISCOUNT_DEPTH: (1 - panel[Col.PRICE] / panel[Col.BASE_PRICE]).clip(0, 1),
            }
        )


@FEATURE_STEPS.register
class FourierSeasonality(FeatureStep):
    """Yearly seasonality as sin/cos pairs of day-of-year.

    `column_names()` lists the generated columns for model formulas.
    """

    name = "fourier"
    SIN: ClassVar[str] = "sin_{k}"
    COS: ClassVar[str] = "cos_{k}"

    def __init__(self, order: int) -> None:
        self._order = order

    @classmethod
    def from_config(cls, cfg: Config) -> FourierSeasonality:
        """Read the Fourier order from config."""
        return cls(cfg.features.fourier_order)

    def column_names(self) -> list[str]:
        """sin_1, cos_1, ..., sin_K, cos_K."""
        return [t.format(k=k) for k in range(1, self._order + 1) for t in (self.SIN, self.COS)]

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Add 2 * order seasonality columns."""
        angle = 2 * np.pi * panel[Col.WEEK].dt.day_of_year.to_numpy() / DAYS_PER_YEAR
        terms = {}
        for k in range(1, self._order + 1):
            terms[self.SIN.format(k=k)] = np.sin(k * angle)
            terms[self.COS.format(k=k)] = np.cos(k * angle)
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
