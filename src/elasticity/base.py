"""Shared building blocks for every elasticity model.

DesignSpec defines the log-log specification (two price terms plus controls);
PriceCentering builds the slope regressors; ElasticityModel is the base each
model registers under ELASTICITY_MODELS; HoldoutScorer computes test WAPE.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from src.config import Config, ElasticityConfig, ModelName
from src.schema import ALL_ESTIMATE_COLS, PRICE_TERMS, Col, Family, driver_columns
from src.utils import AppLogger, DataValidationError, Registry

logger = AppLogger.get_logger(__name__)

# Slope regressors: centred log base price (regular) and log(price / base) (promo).
SLOPE_COLS: dict[Family, str] = {
    Family.REGULAR: "ln_base_price_c",
    Family.PROMO: Col.LN_PROMO_PRICE,
}
ESTIMATE_FRAME_COLS: tuple[str, ...] = (Col.UPC, *ALL_ESTIMATE_COLS)


@dataclass(frozen=True)
class DesignSpec:
    """ln_units ~ regular-price slope + promo-price slope + controls.

    ln_price = ln_base_price + ln(price/base): the first slope is the regular-price
    elasticity (for the optimizer), the second the response to temporary discounts.
    """

    controls: tuple[str, ...]
    response: Col = Col.LN_UNITS

    @classmethod
    def from_config(cls, cfg: Config) -> DesignSpec:
        """Shared drivers (minus the two price terms) plus the residual trend."""
        drivers = driver_columns(cfg.features.fourier_order, cfg.features.pantry_lags)
        price_terms = set(PRICE_TERMS.values())
        return cls(controls=(*[d for d in drivers if d not in price_terms], Col.LN_TREND))

    def active_controls(self, frame: pd.DataFrame) -> list[str]:
        """Controls present in frame with non-zero variance (constant ones are not identified)."""
        present = [c for c in self.controls if c in frame.columns]
        return [c for c in present if frame[c].std() > 0]


class PriceCentering:
    """Builds the slope regressors: base price demeaned per product, promo term as is.

    Centring makes intercepts the product's typical level at its usual regular price.
    """

    def __init__(self) -> None:
        self._means: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> PriceCentering:
        """Learn per-product mean log base price from training rows."""
        self._means = train.groupby(Col.UPC)[Col.LN_BASE_PRICE].mean()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Slope regressors (columns SLOPE_COLS); NaN regular term for unseen products."""
        if self._means is None:
            raise DataValidationError("PriceCentering.transform called before fit")
        centred = frame[Col.LN_BASE_PRICE] - frame[Col.UPC].map(self._means)
        slopes = {
            SLOPE_COLS[Family.REGULAR]: centred,
            SLOPE_COLS[Family.PROMO]: frame[Col.LN_PROMO_PRICE],
        }
        return pd.DataFrame(slopes, index=frame.index).astype(float)


class ElasticityModel(ABC):
    """One way of estimating per-product regular and promo elasticities.

    `fit` returns one row per product (UPC + ALL_ESTIMATE_COLS); `predict` returns
    ln_units for rows of fitted products (NaN otherwise).
    """

    name: ClassVar[ModelName]

    def __init__(self, spec: DesignSpec, settings: ElasticityConfig) -> None:
        self._spec = spec
        self._settings = settings
        self._centering = PriceCentering()

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> pd.DataFrame:
        """Fit on training rows; return UPC + estimate and interval columns for both families."""

    @abstractmethod
    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Predicted ln_units aligned to frame's index."""

    def contribution(self, frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
        """Log-scale effect of `columns`: prediction minus prediction with them set to zero.

        Model-agnostic decomposition (equals sum(coef * x) for linear predictors).
        """
        present = [c for c in columns if c in frame.columns]
        switched_off = frame.assign(**{c: 0.0 for c in present})
        return self.predict(frame) - self.predict(switched_off)

    def save_artifacts(self, directory: Path) -> None:
        """Persist model-specific artifacts; the default model has none to write."""
        logger.debug("No artifacts", extra={"model": self.name.value, "dir": str(directory)})

    @property
    def interval_alpha(self) -> float:
        """Two-sided alpha matching the configured interval width (e.g. 0.06 for 94%)."""
        return 1 - self._settings.bayes.hdi_prob


ELASTICITY_MODELS: Registry[ModelName, type[ElasticityModel]] = Registry("name")


@dataclass(frozen=True)
class HoldoutScore:
    """Holdout accuracy of one model on the units scale."""

    wape: float
    n_rows: int


class HoldoutScorer:
    """Weighted absolute percentage error of exp(predicted ln_units) on test rows.

    Rows a model cannot predict (unfitted products) are excluded and counted.
    """

    def score(self, model: ElasticityModel, test: pd.DataFrame) -> HoldoutScore:
        """WAPE = sum|units - pred| / sum(units) over predictable test rows."""
        predicted = np.exp(model.predict(test))
        ok = predicted.notna()
        actual = test.loc[ok, Col.UNITS]
        wape = float(np.abs(actual - predicted[ok]).sum() / actual.sum()) if ok.any() else np.nan
        return HoldoutScore(wape=wape, n_rows=int(ok.sum()))
