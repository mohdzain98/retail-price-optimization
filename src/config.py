"""Typed configuration and project paths.

Loads config/config.yaml into validated pydantic models so every stage reads
tunables from one place. Also exposes the canonical project directories used
by the pipeline, notebooks and Streamlit app.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator, model_validator

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"


class SheetsConfig(BaseModel):
    """Sheet names inside the raw Breakfast at the Frat workbook.

    Defaults follow the public release; verify against the downloaded file.
    """

    transactions: str
    products: str
    stores: str


class DataConfig(BaseModel):
    """Location and layout of the raw dataset.

    `grain` selects whether models run on product-week or store-product-week.
    """

    raw_file: str
    sheets: SheetsConfig
    header_row: int = 0
    grain: Literal["product_week", "store_product_week"] = "product_week"

    @property
    def raw_path(self) -> Path:
        """Absolute path of the raw workbook under data/raw."""
        return DATA_RAW / self.raw_file


class FiltersConfig(BaseModel):
    """Eligibility rules for products to get their own elasticity.

    Products failing these are kept but receive an imputed elasticity.
    """

    min_weeks: int
    min_price_variation: float


class SplitConfig(BaseModel):
    """Time-based train/holdout split settings.

    The last `holdout_weeks` weeks of each series are held out.
    """

    holdout_weeks: int


class FeaturesConfig(BaseModel):
    """Feature engineering settings for seasonality and holidays.

    Fourier order controls how many sin/cos pairs model yearly seasonality.
    """

    fourier_order: int
    holiday_country: str


class TrendConfig(BaseModel):
    """LOESS trend smoothing settings.

    `lowess_frac` is the fraction of points used for each local fit.
    """

    lowess_frac: float


class BayesConfig(BaseModel):
    """PyMC sampler settings for the hierarchical elasticity model.

    `hdi_prob` sets the width of reported credible intervals.
    """

    draws: int
    tune: int
    chains: int
    target_accept: float
    seed: int
    hdi_prob: float


class ElasticityConfig(BaseModel):
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


class CostConfig(BaseModel):
    """Assumed margins used to derive unit cost from base price.

    The dataset has no cost column, so cost = (1 - margin) * base_price.
    """

    default_margin: float
    margin_by_category: dict[str, float] = {}

    def margin_for(self, category: str) -> float:
        """Margin for a category, falling back to the default."""
        return self.margin_by_category.get(category, self.default_margin)


class OptimizationConfig(BaseModel):
    """Objective, demand form, constraints and NLopt solver settings.

    Price bounds are multipliers of the current price (e.g. 0.8-1.2).
    """

    objective: Literal["profit", "revenue", "weighted"]
    profit_weight: float
    demand_form: Literal["constant_elasticity", "linear"]
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


class ForecastConfig(BaseModel):
    """Forecast horizon for baseline projection.

    Measured in weeks from the last observed week.
    """

    horizon_weeks: int


class PromoConfig(BaseModel):
    """High-level promo budget allocation settings.

    Fixed costs per tactic are assumptions; the dataset has no trade spend.
    """

    budget: float
    tactics: list[Literal["tpr", "feature", "display"]]
    fixed_cost: dict[str, float]
    max_promos_per_sku: int
    min_gap_weeks: int


class Config(BaseModel):
    """Root configuration object mirroring config/config.yaml.

    Obtain it via `load_config()` rather than constructing it directly.
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


@lru_cache(maxsize=4)
def load_config(path: Path = CONFIG_PATH) -> Config:
    """Load and validate the YAML config, cached per path.

    Raises pydantic.ValidationError if any field is missing or out of range.
    """
    with open(path) as f:
        return Config.model_validate(yaml.safe_load(f))
