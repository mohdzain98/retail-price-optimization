"""Turns raw transactions into the modelling panel.

TransactionCleaner repairs rows, an Aggregator (registered per Grain) builds the
panel grain, ProductEligibility applies registered ProductFilters, and
PanelBuilder composes the three and joins product and store attributes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pandas as pd

from src.config import FiltersConfig, Grain
from src.prep.loader import RawData
from src.schema import PROMO_COLS, SUM_COLS, Col, FilterReason
from src.utils import AppLogger, Registry

logger = AppLogger.get_logger(__name__)

_BASE_X_UNITS = "base_x_units"
_P_MIN, _P_MAX = "p_min", "p_max"


class TransactionCleaner:
    """Drops invalid transaction rows and repairs base price.

    Base price is filled within store-UPC, then from price, and floored at price
    (a promo can lower the price below base, never raise it above).
    """

    def clean(self, tx: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned copy; logs how many rows were dropped."""
        valid = tx[(tx[Col.UNITS] > 0) & (tx[Col.PRICE] > 0)]
        ordered = valid.sort_values([Col.STORE, Col.UPC, Col.WEEK])
        by_series = ordered.groupby([Col.STORE, Col.UPC])[Col.BASE_PRICE]
        base = by_series.ffill().groupby([ordered[Col.STORE], ordered[Col.UPC]]).bfill()
        base = np.maximum(base.fillna(ordered[Col.PRICE]), ordered[Col.PRICE])
        promos = {c: ordered[c].fillna(0).astype(int) for c in PROMO_COLS}
        logger.info(
            "Cleaned transactions", extra={"rows_in": len(tx), "dropped": len(tx) - len(ordered)}
        )
        return ordered.assign(**{Col.BASE_PRICE: base}, **promos).reset_index(drop=True)


class Aggregator(ABC):
    """Builds the panel at one grain from cleaned transactions.

    Subclasses declare their grain and key columns and register in AGGREGATORS.
    """

    grain: ClassVar[Grain]
    keys: ClassVar[tuple[Col, ...]]

    @abstractmethod
    def aggregate(self, tx: pd.DataFrame) -> pd.DataFrame:
        """Return one row per key with units, price, base price and promo columns."""


AGGREGATORS: Registry[Grain, type[Aggregator]] = Registry("grain")


@AGGREGATORS.register
class ProductWeekAggregator(Aggregator):
    """Sums units across stores; price = spend / units (unit-weighted).

    Promo columns become the share of selling stores running each tactic.
    """

    grain = Grain.PRODUCT_WEEK
    keys = (Col.UPC, Col.WEEK)

    def aggregate(self, tx: pd.DataFrame) -> pd.DataFrame:
        """Aggregate store rows to product-week."""
        weighted = tx.assign(**{_BASE_X_UNITS: tx[Col.BASE_PRICE] * tx[Col.UNITS]})
        out = weighted.groupby(list(self.keys), as_index=False).agg(
            **{c: (c, "sum") for c in SUM_COLS},
            **{_BASE_X_UNITS: (_BASE_X_UNITS, "sum")},
            **{c: (c, "mean") for c in PROMO_COLS},
            **{Col.N_STORES: (Col.STORE, "nunique")},
        )
        price = out[Col.SPEND] / out[Col.UNITS]
        base = np.maximum(out[_BASE_X_UNITS] / out[Col.UNITS], price)
        return out.drop(columns=_BASE_X_UNITS).assign(**{Col.PRICE: price, Col.BASE_PRICE: base})


@AGGREGATORS.register
class StoreProductWeekAggregator(Aggregator):
    """Keeps the native store x product x week grain (one store per row)."""

    grain = Grain.STORE_PRODUCT_WEEK
    keys = (Col.STORE, Col.UPC, Col.WEEK)

    def aggregate(self, tx: pd.DataFrame) -> pd.DataFrame:
        """Return transactions unchanged apart from an n_stores column of 1."""
        return tx.assign(**{Col.N_STORES: 1})


class ProductFilter(ABC):
    """One eligibility rule over per-product stats (n_weeks, price_variation).

    Registered filters run in registration order; the first failure is the reason.
    """

    reason: ClassVar[FilterReason]

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: FiltersConfig) -> ProductFilter:
        """Build the filter from the filters config."""

    @abstractmethod
    def fails(self, stats: pd.DataFrame) -> pd.Series:
        """Boolean Series (indexed by UPC): True where the product fails."""


PRODUCT_FILTERS: Registry[FilterReason, type[ProductFilter]] = Registry("reason")


@PRODUCT_FILTERS.register
class MinWeeksFilter(ProductFilter):
    """Fails products observed in fewer than `min_weeks` distinct weeks."""

    reason = FilterReason.FEW_WEEKS

    def __init__(self, min_weeks: int) -> None:
        self._min_weeks = min_weeks

    @classmethod
    def from_config(cls, cfg: FiltersConfig) -> MinWeeksFilter:
        """Read min_weeks from config."""
        return cls(cfg.min_weeks)

    def fails(self, stats: pd.DataFrame) -> pd.Series:
        """True where n_weeks < min_weeks."""
        return stats[Col.N_WEEKS] < self._min_weeks


@PRODUCT_FILTERS.register
class PriceVariationFilter(ProductFilter):
    """Fails products whose max/min price spread is below the threshold."""

    reason = FilterReason.LOW_PRICE_VARIATION

    def __init__(self, min_variation: float) -> None:
        self._min_variation = min_variation

    @classmethod
    def from_config(cls, cfg: FiltersConfig) -> PriceVariationFilter:
        """Read min_price_variation from config."""
        return cls(cfg.min_price_variation)

    def fails(self, stats: pd.DataFrame) -> pd.Series:
        """True where price_variation < threshold."""
        return stats[Col.PRICE_VARIATION] < self._min_variation


class ProductEligibility:
    """Flags products eligible for their own elasticity estimate.

    Adds `eligible` and `filter_reason`; ineligible products are kept for imputation.
    """

    def __init__(self, filters: Sequence[ProductFilter]) -> None:
        self._filters = list(filters)

    def flag(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return panel with eligibility columns merged on UPC."""
        stats = self._product_stats(panel)
        reason = pd.Series(FilterReason.NONE.value, index=stats.index)
        for rule in reversed(self._filters):  # reversed so the first rule's reason wins
            reason = reason.mask(rule.fails(stats), rule.reason.value)
        flags = pd.DataFrame({Col.FILTER_REASON: reason, Col.ELIGIBLE: reason == FilterReason.NONE})
        logger.info(
            "Applied product filters",
            extra={
                "eligible": int(flags[Col.ELIGIBLE].sum()),
                "ineligible": reason[reason != ""].value_counts().to_dict(),
            },
        )
        return panel.merge(flags, left_on=Col.UPC, right_index=True)

    @staticmethod
    def _product_stats(panel: pd.DataFrame) -> pd.DataFrame:
        """Per-UPC n_weeks and price_variation (max/min - 1)."""
        stats = panel.groupby(Col.UPC).agg(
            **{
                Col.N_WEEKS: (Col.WEEK, "nunique"),
                _P_MIN: (Col.PRICE, "min"),
                _P_MAX: (Col.PRICE, "max"),
            }
        )
        return stats.assign(**{Col.PRICE_VARIATION: stats[_P_MAX] / stats[_P_MIN] - 1})


class PanelBuilder:
    """Composes cleaning, aggregation and eligibility into the base panel.

    Joins product attributes always and store attributes when store is a key.
    """

    def __init__(
        self, cleaner: TransactionCleaner, aggregator: Aggregator, eligibility: ProductEligibility
    ) -> None:
        self._cleaner = cleaner
        self._aggregator = aggregator
        self._eligibility = eligibility

    @property
    def keys(self) -> tuple[Col, ...]:
        """Key columns of the panel grain."""
        return self._aggregator.keys

    def build(self, raw: RawData) -> pd.DataFrame:
        """Return the panel sorted by key, without engineered features."""
        panel = self._aggregator.aggregate(self._cleaner.clean(raw.transactions))
        panel = self._eligibility.flag(panel).merge(raw.products, on=Col.UPC, how="left")
        if Col.STORE in self.keys:
            panel = panel.merge(raw.stores, on=Col.STORE, how="left")
        return panel.sort_values(list(self.keys)).reset_index(drop=True)
