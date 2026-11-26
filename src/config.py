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
    """Seasonality (Fourier order) and holiday calendar country."""

    fourier_order: int
    holiday_country: str


class TrendConfig(_Frozen):
    """LOESS trend window in weeks; kept above 52 so it skips yearly seasonality."""

    lowess_window_weeks: int


class BayesConfig(_Frozen):
    """PyMC sampler settings; `hdi_prob` sets credible-interval width."""

    draws: int
    tune: int
    chains: int
    target_accept: float
    seed: int
    hdi_prob: float


class ElasticityConfig(_Frozen):
    """Elasticity guard-rails and Bayesian model settings.

    Estimates outside `bounds` are replaced with category-level imputation.
    """

    bounds: tuple[float, float]
    bayes: BayesConfig

    @field_validator("bounds")
    @classmethod
    def _negative_ordered(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Require lower < upper < 0, since demand must fall as price rises."""
        lo, hi = v
        if not lo < hi < 0:
            raise ValueError("elasticity bounds must satisfy lower < upper < 0")
        return v


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

    objective: Objective
    profit_weight: float
    demand_form: DemandForm
    price_bounds: tuple[float, float]
    volume_floor: float
    max_avg_price_increase: float
    ladder_min_gap: float
    algorithms: list[str]
    maxeval: int
    xtol_rel: float
    price_endings: list[float]

    @model_validator(mode="after")
    def _check_ranges(self) -> OptimizationConfig:
        """Validate bound ordering and weight range."""
        lo, hi = self.price_bounds
        if not 0 < lo < 1 < hi:
            raise ValueError("price_bounds must satisfy 0 < lower < 1 < upper")
        if not 0 <= self.profit_weight <= 1:
            raise ValueError("profit_weight must be in [0, 1]")
        return self


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
