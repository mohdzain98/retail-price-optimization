"""Tests for src.prep.panel: cleaning, aggregation, filters and panel building.

Checks invariants downstream stages rely on: no lost units, unit-weighted
prices, base price >= price, and correct eligibility reasons per product.
One test class per production class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Grain
from src.prep.loader import RawData
from src.prep.panel import (
    AGGREGATORS,
    MinWeeksFilter,
    PanelBuilder,
    PriceVariationFilter,
    ProductEligibility,
    ProductWeekAggregator,
    StoreProductWeekAggregator,
    TransactionCleaner,
)
from src.schema import PROMO_COLS, Col, FilterReason
from tests.conftest import SyntheticFrat


def _stats(n_weeks: list[int], variation: list[float]) -> pd.DataFrame:
    """Per-product stats frame as ProductEligibility builds it."""
    return pd.DataFrame({Col.N_WEEKS: n_weeks, Col.PRICE_VARIATION: variation})


class TestTransactionCleaner:
    """TransactionCleaner drops bad rows and repairs base price."""

    def test_drops_bad_rows_and_floors_base(self, raw: RawData) -> None:
        """Zero units / missing price rows go; base price is filled and >= price."""
        tx = TransactionCleaner().clean(raw.transactions)
        assert len(tx) == len(raw.transactions) - 2
        assert (tx[Col.UNITS] > 0).all() and tx[Col.PRICE].notna().all()
        assert tx[Col.BASE_PRICE].notna().all()
        assert (tx[Col.BASE_PRICE] >= tx[Col.PRICE]).all()

    def test_does_not_mutate_input(self, raw: RawData) -> None:
        """The input frame is unchanged."""
        before = raw.transactions.copy()
        TransactionCleaner().clean(raw.transactions)
        pd.testing.assert_frame_equal(before, raw.transactions)


class TestProductWeekAggregator:
    """ProductWeekAggregator sums across stores with weighted prices."""

    def test_preserves_units_and_weights_price(self, raw: RawData) -> None:
        """Total units match; price = spend / units; promo shares in [0, 1]."""
        tx = TransactionCleaner().clean(raw.transactions)
        agg = ProductWeekAggregator().aggregate(tx)
        assert agg[Col.UNITS].sum() == tx[Col.UNITS].sum()
        np.testing.assert_allclose(agg[Col.PRICE], agg[Col.SPEND] / agg[Col.UNITS])
        assert agg[list(PROMO_COLS)].le(1).all().all()
        assert not agg.duplicated([Col.UPC, Col.WEEK]).any()

    def test_registered_for_grain(self) -> None:
        """The registry resolves the product-week grain to this class."""
        assert AGGREGATORS.get(Grain.PRODUCT_WEEK) is ProductWeekAggregator


class TestStoreProductWeekAggregator:
    """StoreProductWeekAggregator keeps the native grain."""

    def test_keeps_rows(self, raw: RawData) -> None:
        """Row count is unchanged and n_stores is 1."""
        tx = TransactionCleaner().clean(raw.transactions)
        out = StoreProductWeekAggregator().aggregate(tx)
        assert len(out) == len(tx) and (out[Col.N_STORES] == 1).all()


class TestMinWeeksFilter:
    """MinWeeksFilter fails products with too few weeks."""

    def test_threshold(self) -> None:
        """Strictly below the minimum fails."""
        assert MinWeeksFilter(52).fails(_stats([51, 52], [0.1, 0.1])).tolist() == [True, False]


class TestPriceVariationFilter:
    """PriceVariationFilter fails flat-priced products."""

    def test_threshold(self) -> None:
        """Strictly below the variation threshold fails."""
        fails = PriceVariationFilter(0.05).fails(_stats([60, 60], [0.0, 0.05]))
        assert fails.tolist() == [True, False]


class TestProductEligibility:
    """ProductEligibility applies filters in order; first failure wins."""

    def test_flags_flat_price_product(self, raw: RawData) -> None:
        """Only the flat-priced UPC is ineligible, with the right reason."""
        agg = ProductWeekAggregator().aggregate(TransactionCleaner().clean(raw.transactions))
        flags = ProductEligibility([MinWeeksFilter(52), PriceVariationFilter(0.05)]).flag(agg)
        per_upc = flags.drop_duplicates(Col.UPC).set_index(Col.UPC)
        assert (
            per_upc.loc[SyntheticFrat.FLAT_UPC, Col.FILTER_REASON]
            == FilterReason.LOW_PRICE_VARIATION
        )
        assert per_upc.drop(SyntheticFrat.FLAT_UPC)[Col.ELIGIBLE].all()

    def test_first_filter_reason_wins(self, raw: RawData) -> None:
        """When both filters fail, the reason is the first filter's."""
        agg = ProductWeekAggregator().aggregate(TransactionCleaner().clean(raw.transactions))
        flags = ProductEligibility([MinWeeksFilter(1000), PriceVariationFilter(5.0)]).flag(agg)
        assert (flags[Col.FILTER_REASON] == FilterReason.FEW_WEEKS).all()


class TestPanelBuilder:
    """PanelBuilder composes cleaning, aggregation, eligibility and joins."""

    def test_builds_sorted_panel_with_attributes(self, raw: RawData) -> None:
        """Panel has product attributes, eligibility and is sorted by key."""
        builder = PanelBuilder(
            TransactionCleaner(), ProductWeekAggregator(), ProductEligibility([MinWeeksFilter(52)])
        )
        panel = builder.build(raw)
        assert {Col.CATEGORY, Col.ELIGIBLE, Col.N_STORES} <= set(panel.columns)
        assert panel.equals(panel.sort_values([Col.UPC, Col.WEEK]).reset_index(drop=True))
