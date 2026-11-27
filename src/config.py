"""Typed, frozen configuration and canonical project paths.

Loads config/config.yaml into validated pydantic models (`Config.load()`), so
every stage reads tunables from one place. Fixed choices are Enums. Paths are
module constants that composition roots pass into the classes that need them.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from src.schema import Col, Tactic
from src.utils.exceptions import MissingFileError

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
PANEL_PATH = DATA_PROCESSED / "panel.parquet"
ELASTICITY_PATH = DATA_PROCESSED / "elasticities.parquet"
ELASTICITY_ALL_MODELS_PATH = DATA_PROCESSED / "elasticities_all_models.parquet"
MODEL_COMPARISON_PATH = DATA_PROCESSED / "model_comparison.parquet"
ELASTICITY_MODELS_PATH = DATA_PROCESSED / "elasticity_models.pkl"
BASELINE_PATH = DATA_PROCESSED / "baseline.parquet"
BASELINE_SUMMARY_PATH = DATA_PROCESSED / "baseline_summary.parquet"
ELASTICITY_DRAWS_PATH = DATA_PROCESSED / "elasticity_draws.parquet"
RECOMMENDATIONS_PATH = DATA_PROCESSED / "price_recommendations.parquet"
OPTIMIZATION_SUMMARY_PATH = DATA_PROCESSED / "optimization_summary.parquet"
FIGURES = ROOT / "reports" / "figures"


class Grain(StrEnum):
    """Panel granularity the models run on."""

    PRODUCT_WEEK = "product_week"
    STORE_PRODUCT_WEEK = "store_product_week"


class Sheet(StrEnum):
    """Logical tables inside the raw workbook."""

    TRANSACTIONS = "transactions"
    PRODUCTS = "products"
    STORES = "stores"


class Objective(StrEnum):
    """What the price optimizer maximizes."""

    PROFIT = "profit"
    REVENUE = "revenue"
    WEIGHTED = "weighted"


class DemandForm(StrEnum):
    """Demand curve used inside the optimizer."""

    CONSTANT_ELASTICITY = "constant_elasticity"
    LINEAR = "linear"


class ModelName(StrEnum):
    """Registered elasticity models, from simplest to most advanced."""

    OLS_PER_PRODUCT = "ols_per_product"
    POOLED_OLS = "pooled_ols"
    MIXED_EFFECTS = "mixed_effects"
    BAYES_HIERARCHICAL = "bayes_hierarchical"


class Algorithm(StrEnum):
    """NLopt algorithms compared for price optimization (names match nlopt constants)."""

    LD_SLSQP = "LD_SLSQP"  # local, gradient-based, handles nonlinear constraints
    LN_COBYLA = "LN_COBYLA"  # local, derivative-free
    GN_ISRES = "GN_ISRES"  # global, evolutionary, derivative-free


class _Frozen(BaseModel):
    """Base for all config models: immutable after validation."""

    model_config = ConfigDict(frozen=True)


class DataConfig(_Frozen):
    """Location and layout of the raw dataset.

    `columns` maps raw column names to canonical `Col` names, per sheet.
    """

    raw_file: str
    sheets: dict[Sheet, str]
    header_row: int | None = None
    grain: Grain = Grain.PRODUCT_WEEK
    columns: dict[Sheet, dict[str, Col]]

    @model_validator(mode="after")
    def _covers_all_sheets(self) -> DataConfig:
        """Require a sheet name and a column mapping for every Sheet."""
        for name, mapping in (("sheets", self.sheets), ("columns", self.columns)):
            missing = set(Sheet) - set(mapping)
            if missing:
                raise ValueError(f"data.{name} is missing {sorted(missing)}")
        return self

    @property
    def raw_path(self) -> Path:
        """Absolute path of the raw workbook under data/raw."""
        return DATA_RAW / self.raw_file


class FiltersConfig(_Frozen):
    """Eligibility rules for products to get their own elasticity.

    Products failing these are kept but receive an imputed elasticity.
    """

    min_weeks: int
    min_price_variation: float


class SplitConfig(_Frozen):
    """Time-based train/holdout split: the last `holdout_weeks` weeks are test."""

    holdout_weeks: int


class FeaturesConfig(_Frozen):
    """Seasonality, holiday calendar, promo-flag thresholds and pantry-loading lags."""

    fourier_order: int
    holiday_country: str
    promo_min_discount: float
    promo_min_tactic_share: float
    pantry_lags: int


class TrendConfig(_Frozen):
    """LOESS trend window in weeks; kept above 52 so it skips yearly seasonality."""

    lowess_window_weeks: int


class BayesConfig(_Frozen):
    """PyMC sampler settings and priors; `hdi_prob` sets interval width.

    Priors: global mean elasticity ~ Normal(prior_mean, prior_sd); category and
    product deviations have HalfNormal(prior_group_sd) scales.
    """

    draws: int
    tune: int
    chains: int
    target_accept: float
    seed: int
    hdi_prob: float
    prior_mean: float
    prior_sd: float
    promo_prior_mean: float
    promo_prior_sd: float
    prior_group_sd: float


class QualityConfig(_Frozen):
    """Per-product quality gates (RGMx-style) a model must pass to be used."""

    max_mape: float
    min_r2: float


class ElasticityConfig(_Frozen):
    """Elasticity models, selection priority, quality gates, bounds and imputation.

    Per product, the first model in `model_priority` passing the gates is used;
    otherwise the elasticity is imputed along `imputation_levels`.
    """

    bounds: tuple[float, float]
    promo_bounds: tuple[float, float]
    models: list[ModelName]
    model_priority: list[ModelName]
    quality: QualityConfig
    imputation_levels: list[list[Col]]
    bayes: BayesConfig

    @field_validator("bounds", "promo_bounds")
    @classmethod
    def _negative_ordered(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Require lower < upper < 0, since demand must fall as price rises."""
        lo, hi = v
        if not lo < hi < 0:
            raise ValueError("elasticity bounds must satisfy lower < upper < 0")
        return v

    @model_validator(mode="after")
    def _priority_is_fitted(self) -> ElasticityConfig:
        """Every model in the priority list must be fitted."""
        missing = [m for m in self.model_priority if m not in self.models]
        if missing or not self.model_priority:
            raise ValueError(f"model_priority {missing} not in models {self.models}")
        return self


class CostConfig(_Frozen):
    """Assumed margins: cost = (1 - margin) * base_price (no cost column in data)."""

    default_margin: float
    margin_by_category: dict[str, float] = {}

    def margin_for(self, category: str) -> float:
        """Margin for a category, falling back to the default."""
        return self.margin_by_category.get(category, self.default_margin)


class OptimizationConfig(_Frozen):
    """Objective, demand form, constraints and NLopt solver settings.

    Price bounds are multipliers of the current price (e.g. 0.8-1.2).
    """

    objectives: list[Objective]
    profit_weight: float
    demand_form: DemandForm
    price_bounds: tuple[float, float]
    volume_floor: float
    max_avg_price_increase: float
    ladder_min_gap: float
    algorithms: list[Algorithm]
    maxeval: int
    max_seconds: float
    xtol_rel: float
    price_endings: list[float]
    reference_weeks: int
    annualize_weeks: int
    robustness_draws: int
    seed: int

    @model_validator(mode="after")
    def _check_ranges(self) -> OptimizationConfig:
        """Validate bound ordering and weight range."""
        lo, hi = self.price_bounds
        if not 0 < lo < 1 < hi:
            raise ValueError("price_bounds must satisfy 0 < lower < 1 < upper")
        if not 0 <= self.profit_weight <= 1:
            raise ValueError("profit_weight must be in [0, 1]")
        if not self.objectives:
            raise ValueError("objectives must list at least one objective")
        return self


class BaselineConfig(_Frozen):
    """Top-down baseline and RGMx bad-baseline correction settings.

    Rules flag weeks where the raw baseline is implausible; see src/baseline/rules.py.
    """

    rolling_window: int
    max_base_to_volume: float
    neighbor_drop_ratio: float
    promo_lookback_weeks: int


class ForecastConfig(_Frozen):
    """Forecast horizon in weeks from the last observed week."""

    horizon_weeks: int


class PromoConfig(_Frozen):
    """High-level promo budget settings; tactic fixed costs are assumptions."""

    budget: float
    tactics: list[Tactic]
    fixed_cost: dict[Tactic, float]
    max_promos_per_sku: int
    min_gap_weeks: int


_CACHE: dict[Path, Config] = {}


class Config(_Frozen):
    """Root configuration mirroring config/config.yaml.

    Obtain it with `Config.load()`; instances are cached per file path.
    """

    data: DataConfig
    filters: FiltersConfig
    split: SplitConfig
    features: FeaturesConfig
    trend: TrendConfig
    elasticity: ElasticityConfig
    cost: CostConfig
    optimization: OptimizationConfig
    baseline: BaselineConfig
    forecast: ForecastConfig
    promo: PromoConfig

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> Config:
        """Load and validate the YAML config, cached per path.

        Raises MissingFileError if the file is absent, ValidationError if invalid.
        """
        if path not in _CACHE:
            if not path.exists():
                raise MissingFileError(f"Config file not found: {path}")
            _CACHE[path] = cls.model_validate(yaml.safe_load(path.read_text()))
        return _CACHE[path]
