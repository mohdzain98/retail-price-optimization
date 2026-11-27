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
    CURRENT_PRICE = "current_price"
    BASE_VOLUME = "base_volume_weekly"
    UNIT_COST = "unit_cost"
    SIZE_VALUE = "size_value"
    SIZE_UNIT = "size_unit"
    PRICE_RATIO = "price_ratio"
    LINEAR_FORMULA_PRICE = "linear_formula_price"
    ROBUST_P05 = "optimal_price_p05"
    ROBUST_P50 = "optimal_price_p50"
    ROBUST_P95 = "optimal_price_p95"
    SCENARIO = "scenario"
    OBJECTIVE_VALUE = "objective_value"
    EVALUATIONS = "evaluations"
    STATUS = "status"
    MAX_VIOLATION = "max_constraint_violation"
    OBJECTIVE = "objective"
    EFFECTIVE_COST = "effective_cost"
    OWN_OPTIMAL_PRICE = "own_optimal_price"
    EXPECTED_DIRECTION = "expected_direction"
    ACTUAL_DIRECTION = "actual_direction"
    RULE_STATUS = "rule_status"
    SAFEGUARD_APPLIED = "safeguard_applied"
    MODEL = "model"
    ELASTICITY_SOURCE = "elasticity_source"
    IMPUTE_REASON = "impute_reason"
    PROMO_IMPUTE_REASON = "promo_impute_reason"
    WAPE = "holdout_wape"
    N_PRODUCTS = "n_products"
    FIT_SECONDS = "fit_seconds"
    MEDIAN_ELASTICITY = "median_elasticity"
    SHARE_IN_BOUNDS = "share_in_bounds"
    HORIZON_WEEK = "horizon_week"
    SEASONAL_INDEX = "seasonal_index"
    REGULAR_PRICE = "regular_price"
    FORECAST_CURRENT = "forecast_baseline_current_prices"
    FORECAST_BASELINE = "forecast_baseline_units"
    TACTIC = "tactic"
    PROMO_PRICE = "promo_price"
    LIFT = "lift"
    PROMO_UNITS = "promo_units"
    DIP_UNITS = "pantry_dip_units"
    INCREMENTAL_REVENUE = "incremental_revenue"
    INCREMENTAL_PROFIT = "incremental_profit"
    FIXED_COST = "fixed_cost"
    TRADE_SPEND = "trade_spend"
    GAIN = "gain"
    ROI = "roi"
    SELECTED = "selected"
    PLANNER = "planner"
    BUDGET = "budget"
    N_PROMOS = "n_promos"
    MARGINAL_RETURN = "marginal_return"
    RECOMMENDED_BUDGET = "recommended_budget"
    METHOD = "method"
    N_ROWS = "n_rows"
    BIAS = "holdout_bias"
    FORECAST_METHOD = "forecast_method"
    PLAN_UNITS = "plan_units"
    VOLUME = "volume"
    REVENUE = "revenue"
    PROFIT = "profit"


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
    """Top-down baseline effect terms: removed from actual log volume."""

    PRICE = "price"
    PROMO = "promo"
    PANTRY = "pantry"
    ERROR = "error"


class BaselineFix(StrEnum):
    """How a week's baseline was finalised (bad-baseline correction)."""

    NONE = "raw"
    ACTUAL = "actual_volume"
    ROLLING = "rolling_mean"


class Scenario(StrEnum):
    """Price scenarios scored by the optimization stage."""

    CURRENT = "current"
    OPTIMAL = "optimal"
    RECOMMENDED = "recommended"  # optimal rounded to price points
    LINEAR_FORMULA = "linear_formula"  # linear-demand optimum scored on the true curve


class Direction(StrEnum):
    """Price move direction relative to the current price."""

    RAISE = "raise"
    CUT = "cut"
    HOLD = "hold"


class RuleStatus(StrEnum):
    """Whether a recommendation follows the objective's own-optimum direction."""

    FOLLOWS = "follows_rule"
    CONSTRAINT_OVERRIDE = "constraint_override"


class Metric(StrEnum):
    """Per-product weekly KPIs for a price scenario."""

    PRICE = "price"
    VOLUME = "volume"
    REVENUE = "revenue"
    PROFIT = "profit"


class Tactic(StrEnum):
    """Promo tactics present in the dataset; values match their panel columns."""

    FEATURE = "feature"
    DISPLAY = "display"
    TPR = "tpr"
    FEATURE_DISPLAY = "feature_display"


class Planner(StrEnum):
    """Promo calendar planners: exact MILP and the greedy ROI-ranking benchmark."""

    MILP = "milp"
    GREEDY = "greedy"


class PromoScenario(StrEnum):
    """Horizon scenarios of the promo stage; every plan uses the recommended prices."""

    CURRENT_NO_PROMO = "current_prices_no_promo"
    RECOMMENDED_NO_PROMO = "recommended_prices_no_promo"
    LAST_YEAR_PLAN = "last_year_calendar"  # business as usual
    MILP_SAME_SPEND = "milp_at_last_year_spend"
    GREEDY_PLAN = "greedy_plan"
    MILP_PLAN = "milp_plan"


class ForecastMethod(StrEnum):
    """Baseline forecast methods compared in the backtest."""

    NAIVE = "naive_flat"
    SEASONAL = "seasonal_index"
    SARIMAX = "sarimax"
    XGBOOST = "xgboost"


PROMO_COLS: tuple[Col, ...] = (Col.FEATURE, Col.DISPLAY, Col.TPR)
TACTIC_COLS: tuple[Col, ...] = (Col.FEATURE, Col.DISPLAY, Col.FEATURE_DISPLAY, Col.TPR)
SUM_COLS: tuple[Col, ...] = (Col.UNITS, Col.SPEND, Col.VISITS, Col.HHS)
HOLIDAY_COLS: tuple[Col, ...] = (Col.HOLIDAY_WEEK, Col.SUPER_BOWL, Col.THANKSGIVING, Col.CHRISTMAS)
# Panel columns switched on (set to 1) by each tactic; feature+display also has its own term.
TACTIC_TERMS: dict[Tactic, tuple[Col, ...]] = {
    Tactic.TPR: (Col.TPR,),
    Tactic.FEATURE: (Col.FEATURE,),
    Tactic.DISPLAY: (Col.DISPLAY,),
    Tactic.FEATURE_DISPLAY: (Col.FEATURE, Col.DISPLAY, Col.FEATURE_DISPLAY),
}
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


def kpi_col(scenario: Scenario, metric: Metric) -> str:
    """KPI column name, e.g. optimal_profit (single definition)."""
    return f"{scenario.value}_{metric.value}"


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
