"""Single source of truth for column names and fixed data-level choices.

Every DataFrame column the pipeline creates or reads is a `Col` member, so a
rename happens in one place. Also holds enums for split labels, filter reasons
and promo tactics, plus the column groups that several modules share.
"""

from __future__ import annotations

from enum import StrEnum


class Col(StrEnum):
    """Canonical column names across raw tables, the panel and model outputs.

    StrEnum members behave as plain strings in pandas and parquet.
    """

    WEEK = "week"
    STORE = "store"
    UPC = "upc"
    UNITS = "units"
    VISITS = "visits"
    HHS = "hhs"
    SPEND = "spend"
    PRICE = "price"
    BASE_PRICE = "base_price"
    FEATURE = "feature"
    DISPLAY = "display"
    TPR = "tpr"
    DESCRIPTION = "description"
    MANUFACTURER = "manufacturer"
    CATEGORY = "category"
    SUB_CATEGORY = "sub_category"
    PRODUCT_SIZE = "product_size"
    STORE_NAME = "store_name"
    STATE = "state"
    SEGMENT = "segment"
    N_STORES = "n_stores"
    N_WEEKS = "n_weeks"
    PRICE_VARIATION = "price_variation"
    ELIGIBLE = "eligible"
    FILTER_REASON = "filter_reason"
    LN_UNITS = "ln_units"
    LN_PRICE = "ln_price"
    LN_BASE_PRICE = "ln_base_price"
    DISCOUNT_DEPTH = "discount_depth"
    HOLIDAY_WEEK = "holiday_week"
    SUPER_BOWL = "super_bowl"
    THANKSGIVING = "thanksgiving"
    CHRISTMAS = "christmas"
    T = "t"
    SPLIT = "split"
    LN_TREND = "ln_trend"


class Split(StrEnum):
    """Time-split labels written to `Col.SPLIT`."""

    TRAIN = "train"
    TEST = "test"


class FilterReason(StrEnum):
    """Why a product is ineligible for its own elasticity (empty = eligible)."""

    NONE = ""
    FEW_WEEKS = "few_weeks"
    LOW_PRICE_VARIATION = "low_price_variation"


class Tactic(StrEnum):
    """Promo tactics present in the dataset; values match their panel columns."""

    FEATURE = "feature"
    DISPLAY = "display"
    TPR = "tpr"


PROMO_COLS: tuple[Col, ...] = (Col.FEATURE, Col.DISPLAY, Col.TPR)
SUM_COLS: tuple[Col, ...] = (Col.UNITS, Col.SPEND, Col.VISITS, Col.HHS)
