"""Per-product inputs to the price optimizer, built from the baseline and elasticities.

For each product: current regular price P0 (median base price over the last N
weeks), weekly base volume B (mean baseline over the same weeks), regular-price
elasticity with its interval, unit cost from the category margin, and pack size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np
import pandas as pd

from src.config import CostConfig
from src.schema import Col
from src.utils import AppLogger, DataValidationError

logger = AppLogger.get_logger(__name__)

PRODUCT_ATTRS: tuple[Col, ...] = (
    Col.UPC, Col.DESCRIPTION, Col.PRODUCT_SIZE, Col.MANUFACTURER, Col.CATEGORY, Col.SUB_CATEGORY,
)  # fmt: skip
ELASTICITY_ATTRS: tuple[Col, ...] = (
    Col.ELASTICITY, Col.ELASTICITY_LOW, Col.ELASTICITY_HIGH, Col.ELASTICITY_SOURCE, Col.MODEL,
)  # fmt: skip


@dataclass(frozen=True)
class OptimizationInputs:
    """One row per product with everything the demand curve and objective need.

    Arrays are aligned to `products` row order; use `with_beta` for scenarios.
    """

    products: pd.DataFrame

    @property
    def p0(self) -> np.ndarray:
        """Current regular price per product."""
        return self.products[Col.CURRENT_PRICE].to_numpy(float)

    @property
    def base_volume(self) -> np.ndarray:
        """Weekly baseline volume at the current price."""
        return self.products[Col.BASE_VOLUME].to_numpy(float)

    @property
    def beta(self) -> np.ndarray:
        """Regular-price elasticity per product."""
        return self.products[Col.ELASTICITY].to_numpy(float)

    @property
    def cost(self) -> np.ndarray:
        """Unit cost per product."""
        return self.products[Col.UNIT_COST].to_numpy(float)

    def with_beta(self, beta: np.ndarray) -> OptimizationInputs:
        """Copy with replaced elasticities (for robustness scenarios)."""
        return replace(self, products=self.products.assign(**{Col.ELASTICITY: beta}))


class PackSizeParser:
    """Parses pack sizes such as '12.25 OZ', '1.5 LT', '16 CT' into value and unit.

    Litres become millilitres so all liquids share one unit.
    """

    PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"([\d.]+)\s*(OZ|ML|LT|CT)", re.IGNORECASE)
    TO_ML: ClassVar[float] = 1000.0

    def parse(self, sizes: pd.Series) -> pd.DataFrame:
        """Columns size_value (float, NaN if unparseable) and size_unit."""
        parts = sizes.astype(str).str.extract(self.PATTERN)
        value = pd.to_numeric(parts[0], errors="coerce")
        unit = parts[1].str.upper()
        is_litre = unit == "LT"
        return pd.DataFrame(
            {
                Col.SIZE_VALUE: value.where(~is_litre, value * self.TO_ML),
                Col.SIZE_UNIT: unit.where(~is_litre, "ML"),
            },
            index=sizes.index,
        )


class InputsBuilder:
    """Builds OptimizationInputs from the weekly baseline and the final elasticities.

    Uses the last `reference_weeks` weeks so P0 and B describe the current state.
    """

    def __init__(self, reference_weeks: int, cost: CostConfig, sizes: PackSizeParser) -> None:
        self._weeks = reference_weeks
        self._cost = cost
        self._sizes = sizes

    def build(self, baseline: pd.DataFrame, final: pd.DataFrame) -> OptimizationInputs:
        """One row per product with P0, B, elasticity, unit cost and pack size."""
        cutoff = np.sort(baseline[Col.WEEK].unique())[-self._weeks]
        recent = baseline[baseline[Col.WEEK] >= cutoff]
        state = recent.groupby(Col.UPC).agg(
            **{
                Col.CURRENT_PRICE: (Col.BASE_PRICE, "median"),
                Col.BASE_VOLUME: (Col.BASELINE, "mean"),
            }
        )
        products = (
            final[list(PRODUCT_ATTRS) + list(ELASTICITY_ATTRS)]
            .merge(state, left_on=Col.UPC, right_index=True, how="inner")
            .reset_index(drop=True)
        )
        dropped = sorted(set(final[Col.UPC]) - set(products[Col.UPC]))
        if dropped:
            logger.warning("Products with no recent sales excluded", extra={"upcs": dropped})
        if products[[Col.CURRENT_PRICE, Col.BASE_VOLUME, Col.ELASTICITY]].isna().any().any():
            raise DataValidationError("Missing price, volume or elasticity for some products")
        margin = products[Col.CATEGORY].map(self._cost.margin_for)
        products = products.assign(**{Col.UNIT_COST: (1 - margin) * products[Col.CURRENT_PRICE]})
        products = pd.concat([products, self._sizes.parse(products[Col.PRODUCT_SIZE])], axis=1)
        logger.info(
            "Built optimizer inputs", extra={"products": len(products), "weeks": self._weeks}
        )
        return OptimizationInputs(products)
