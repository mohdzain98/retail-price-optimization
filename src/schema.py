"""Single source of truth for column names and fixed data-level choices.

Every DataFrame column the pipeline creates or reads is a `Col` member (Fourier
names come from `fourier_columns`), so a rename happens in one place. Also holds
enums for splits, filter/impute reasons and tactics, plus shared column groups.
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
    FEATURE_DISPLAY = "feature_display"
    DESCRIPTION = "description"
    MANUFACTURER = "manufacturer"
    CATEGORY = "category"
    SUB_CATEGORY = "sub_category"
    PRODUCT_SIZE = "product_size"
    STORE_NAME = "store_name"
    STATE = "state"
    SEGMENT = "segment"
    N_STORES = "n_stores"
    LN_N_STORES = "ln_n_stores"
    N_WEEKS = "n_weeks"
    PRICE_VARIATION = "price_variation"
    ELIGIBLE = "eligible"
    FILTER_REASON = "filter_reason"
    LN_UNITS = "ln_units"
    LN_PRICE = "ln_price"
    LN_BASE_PRICE = "ln_base_price"
    DISCOUNT_DEPTH = "discount_depth"
    LN_PROMO_PRICE = "ln_promo_price"
    PROMO_FLAG = "promo_flag"
    HOLIDAY_WEEK = "holiday_week"
    SUPER_BOWL = "super_bowl"
    THANKSGIVING = "thanksgiving"
    CHRISTMAS = "christmas"
    T = "t"
    SPLIT = "split"
    LN_TREND = "ln_trend"
    ELASTICITY = "elasticity"
    ELASTICITY_LOW = "elasticity_low"
    ELASTICITY_HIGH = "elasticity_high"
    PROMO_ELASTICITY = "promo_elasticity"
    PROMO_ELASTICITY_LOW = "promo_elasticity_low"
    PROMO_ELASTICITY_HIGH = "promo_elasticity_high"
    PROMO_SOURCE = "promo_source"
    MAPE = "train_mape"
    R2 = "train_r2"
    PASSES_GATES = "passes_gates"
    SHARE_PASSING = "share_passing_gates"
    BASELINE_MODEL = "baseline_model"
    LN_PRED = "ln_pred"
    BASE_RAW = "base_raw"
    BASELINE = "baseline_units"
    BASELINE_FIX = "baseline_fix"
    INCREMENTAL = "incremental_units"
    BASELINE_SHARE = "baseline_share"
    PROMO_WEEK_LIFT = "promo_week_lift"
    MODEL = "model"
    ELASTICITY_SOURCE = "elasticity_source"
    IMPUTE_REASON = "impute_reason"
    PROMO_IMPUTE_REASON = "promo_impute_reason"
    WAPE = "holdout_wape"
    N_PRODUCTS = "n_products"
    FIT_SECONDS = "fit_seconds"
    MEDIAN_ELASTICITY = "median_elasticity"
    SHARE_IN_BOUNDS = "share_in_bounds"


class Split(StrEnum):
    """Time-split labels written to `Col.SPLIT`."""

    TRAIN = "train"
    TEST = "test"


class FilterReason(StrEnum):
    """Why a product is ineligible for its own elasticity (empty = eligible)."""

    NONE = ""
    FEW_WEEKS = "few_weeks"
    LOW_PRICE_VARIATION = "low_price_variation"


class ElasticitySource(StrEnum):
    """Whether a product's final elasticity was estimated or imputed."""

    ESTIMATED = "estimated"
    IMPUTED = "imputed"


class ImputeReason(StrEnum):
    """Why a product's elasticity was imputed (empty = estimated)."""

    NONE = ""
    INELIGIBLE = "ineligible"
    NO_MODEL_PASSED = "no_model_passed_gates"
    NOT_ESTIMATED = "not_estimated"
    OUT_OF_BOUNDS = "out_of_bounds"


class Family(StrEnum):
    """Elasticity families: regular-price (optimizer) and promo-discount (promo planning)."""

    REGULAR = "regular"
    PROMO = "promo"


class EffectTerm(StrEnum):
    """Top-down baseline effect terms (RGMx): removed from actual log volume."""

    PRICE = "price"
    PROMO = "promo"
    PANTRY = "pantry"
    ERROR = "error"


class BaselineFix(StrEnum):
    """How a week's baseline was finalised (bad-baseline correction)."""

    NONE = "raw"
    ACTUAL = "actual_volume"
    ROLLING = "rolling_mean"


class Tactic(StrEnum):
    """Promo tactics present in the dataset; values match their panel columns."""

    FEATURE = "feature"
    DISPLAY = "display"
    TPR = "tpr"
    FEATURE_DISPLAY = "feature_display"


PROMO_COLS: tuple[Col, ...] = (Col.FEATURE, Col.DISPLAY, Col.TPR)
TACTIC_COLS: tuple[Col, ...] = (Col.FEATURE, Col.DISPLAY, Col.FEATURE_DISPLAY, Col.TPR)
SUM_COLS: tuple[Col, ...] = (Col.UNITS, Col.SPEND, Col.VISITS, Col.HHS)
HOLIDAY_COLS: tuple[Col, ...] = (Col.HOLIDAY_WEEK, Col.SUPER_BOWL, Col.THANKSGIVING, Col.CHRISTMAS)
PRICE_TERMS: dict[Family, Col] = {
    Family.REGULAR: Col.LN_BASE_PRICE,
    Family.PROMO: Col.LN_PROMO_PRICE,
}
ESTIMATE_COLS: dict[Family, tuple[Col, Col, Col]] = {
    Family.REGULAR: (Col.ELASTICITY, Col.ELASTICITY_LOW, Col.ELASTICITY_HIGH),
    Family.PROMO: (Col.PROMO_ELASTICITY, Col.PROMO_ELASTICITY_LOW, Col.PROMO_ELASTICITY_HIGH),
}
SOURCE_COLS: dict[Family, tuple[Col, Col]] = {
    Family.REGULAR: (Col.ELASTICITY_SOURCE, Col.IMPUTE_REASON),
    Family.PROMO: (Col.PROMO_SOURCE, Col.PROMO_IMPUTE_REASON),
}
ALL_ESTIMATE_COLS: tuple[Col, ...] = tuple(c for cols in ESTIMATE_COLS.values() for c in cols)
EFFECT_COLS: dict[EffectTerm, str] = {t: f"{t.value}_effect" for t in EffectTerm}
LIFT_COLS: dict[EffectTerm, str] = {t: f"{t.value}_lift" for t in EffectTerm}
_FOURIER_TEMPLATES = ("sin_{k}", "cos_{k}")
_PANTRY_TEMPLATE = "pl{i}"


def fourier_columns(order: int) -> list[str]:
    """Seasonality column names sin_1, cos_1, ..., sin_K, cos_K (single definition)."""
    return [t.format(k=k) for k in range(1, order + 1) for t in _FOURIER_TEMPLATES]


def pantry_columns(lags: int) -> list[str]:
    """Pantry-loading column names pl1..plN (promo flag lagged i weeks)."""
    return [_PANTRY_TEMPLATE.format(i=i) for i in range(1, lags + 1)]


def driver_columns(fourier_order: int, pantry_lags: int) -> list[str]:
    """Demand drivers shared by the residual trend and the elasticity model (excl. trend).

    Price terms first, then promotion, coverage, holidays, pantry loading and seasonality.
    """
    return [
        *PRICE_TERMS.values(), *TACTIC_COLS, Col.LN_N_STORES, *HOLIDAY_COLS,
        *pantry_columns(pantry_lags), *fourier_columns(fourier_order),
    ]  # fmt: skip
