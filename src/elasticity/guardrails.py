"""Turns raw model estimates into the final elasticity table (RGMx-style post-processing).

QualityGates scores each model per product (in-sample MAPE, R^2, bounds);
ModelSelector takes, per product, the first model in priority order that passes;
HierarchicalImputer fills the rest from similar products. Every row is flagged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import ModelName, QualityConfig
from src.elasticity.base import ElasticityModel
from src.schema import ESTIMATE_COLS, SOURCE_COLS, Col, ElasticitySource, Family, ImputeReason
from src.utils import AppLogger, DataValidationError

logger = AppLogger.get_logger(__name__)

PRODUCT_COLS: tuple[Col, ...] = (
    Col.UPC, Col.DESCRIPTION, Col.PRODUCT_SIZE, Col.MANUFACTURER,
    Col.CATEGORY, Col.SUB_CATEGORY, Col.ELIGIBLE,
)  # fmt: skip
_ABS_PCT_ERR, _SQ_RES, _SQ_TOT = "abs_pct_err", "sq_res", "sq_tot"


class QualityGates:
    """Per-product fit quality of one model and whether it passes the gates.

    Gates: in-sample MAPE on units, R^2 on log units within product, regular elasticity in bounds.
    """

    def __init__(self, quality: QualityConfig, bounds: tuple[float, float]) -> None:
        self._quality = quality
        self._low, self._high = bounds

    @property
    def bounds(self) -> tuple[float, float]:
        """(lower, upper) accepted regular elasticity range."""
        return self._low, self._high

    def evaluate(
        self, model: ElasticityModel, estimates: pd.DataFrame, train: pd.DataFrame
    ) -> pd.DataFrame:
        """Estimates plus train MAPE, R^2 and the pass flag, one row per product."""
        per_upc = self._fit_quality(model.predict(train), train)
        scored = estimates.merge(per_upc, left_on=Col.UPC, right_index=True, how="left")
        passes = (
            (scored[Col.MAPE] <= self._quality.max_mape)
            & (scored[Col.R2] >= self._quality.min_r2)
            & scored[Col.ELASTICITY].between(self._low, self._high)
        )
        return scored.assign(**{Col.PASSES_GATES: passes})

    @staticmethod
    def _fit_quality(pred: pd.Series, train: pd.DataFrame) -> pd.DataFrame:
        """Per-UPC MAPE (units) and R^2 (log units, within product)."""
        ln_units, upc = train[Col.LN_UNITS], train[Col.UPC]
        errors = pd.DataFrame(
            {
                Col.UPC: upc,
                _ABS_PCT_ERR: (np.exp(pred) - train[Col.UNITS]).abs() / train[Col.UNITS],
                _SQ_RES: (ln_units - pred) ** 2,
                _SQ_TOT: (ln_units - ln_units.groupby(upc).transform("mean")) ** 2,
            }
        )
        sums = errors.groupby(Col.UPC).agg(
            **{
                Col.MAPE: (_ABS_PCT_ERR, "mean"),
                _SQ_RES: (_SQ_RES, "sum"),
                _SQ_TOT: (_SQ_TOT, "sum"),
            }
        )
        return sums.assign(**{Col.R2: 1 - sums[_SQ_RES] / sums[_SQ_TOT]})[[Col.MAPE, Col.R2]]


class ModelSelector:
    """Picks, per product, the first model in priority order that passed the gates.

    Products no model passes are absent from the result (they get imputed).
    """

    def __init__(self, priority: Sequence[ModelName]) -> None:
        self._priority = list(priority)

    def select(self, scored: Mapping[ModelName, pd.DataFrame]) -> pd.DataFrame:
        """One row per product with its chosen model's estimates and quality."""
        chosen: list[pd.DataFrame] = []
        taken: set = set()
        for name in self._priority:
            frame = scored[name]
            passing = frame[frame[Col.PASSES_GATES] & ~frame[Col.UPC].isin(taken)]
            chosen.append(passing.assign(**{Col.MODEL: name.value}))
            taken.update(passing[Col.UPC])
        selected = pd.concat(chosen, ignore_index=True)
        counts = selected[Col.MODEL].value_counts().to_dict()
        logger.info("Selected models", extra={"by_model": counts})
        return selected.drop(columns=Col.PASSES_GATES)


class HierarchicalImputer:
    """Fills invalid elasticities with the median of valid ones at the closest level.

    Levels run from most to least specific (e.g. sub-category + manufacturer, then
    category), then the global median. Each family is imputed independently.
    """

    def __init__(
        self, levels: Sequence[Sequence[Col]], bounds: Mapping[Family, tuple[float, float]]
    ) -> None:
        self._levels = [list(level) for level in levels]
        self._bounds = dict(bounds)

    def apply(self, selected: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
        """One row per product: final estimates for both families with source and reason."""
        table = products[list(PRODUCT_COLS)].merge(selected, on=Col.UPC, how="left")
        for family in ESTIMATE_COLS:
            table = self._impute(table, family)
        return table

    def _impute(self, table: pd.DataFrame, family: Family) -> pd.DataFrame:
        """Impute one family's estimate and interval; record source and reason."""
        cols = list(ESTIMATE_COLS[family])
        reason = self._reasons(table, family)
        valid = pd.Series(reason == ImputeReason.NONE.value, index=table.index)
        if not valid.any():
            raise DataValidationError(f"No valid {family.value} elasticities to impute from")
        filled = table[cols].where(valid)
        for level in self._levels:
            medians = table[valid].groupby(level)[cols].median().reset_index()
            lookup = table[level].merge(medians, on=level, how="left")[cols]
            filled = filled.fillna(lookup.set_axis(table.index))
        filled = filled.fillna(table.loc[valid, cols].median())
        source_col, reason_col = SOURCE_COLS[family]
        source = np.where(valid, ElasticitySource.ESTIMATED.value, ElasticitySource.IMPUTED.value)
        counts = pd.Series(reason[~valid.to_numpy()]).value_counts().to_dict()
        logger.info("Imputed elasticities", extra={"family": family.value, "imputed": counts})
        return table.assign(
            **{c: filled[c] for c in cols}, **{source_col: source, reason_col: reason}
        )

    def _reasons(self, table: pd.DataFrame, family: Family) -> np.ndarray:
        """First failing rule per product: ineligible, no model passed, missing, out of bounds."""
        estimate = table[ESTIMATE_COLS[family][0]]
        low, high = self._bounds[family]
        conditions = [
            ~table[Col.ELIGIBLE].astype(bool),
            table[Col.MODEL].isna(),
            estimate.isna(),
            ~estimate.between(low, high),
        ]
        reasons = [
            ImputeReason.INELIGIBLE.value,
            ImputeReason.NO_MODEL_PASSED.value,
            ImputeReason.NOT_ESTIMATED.value,
            ImputeReason.OUT_OF_BOUNDS.value,
        ]
        return np.select(conditions, reasons, default=ImputeReason.NONE.value)
