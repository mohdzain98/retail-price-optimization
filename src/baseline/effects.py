"""Top-down effect terms per product-week (RGMx `04_calculate_baseline`).

Each product's volume is decomposed with its selected elasticity model (or the
first model in priority if none passed the gates). The effects removed from actual
log volume are price (promo deviation), promo tactics, pantry loading and error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import ModelName
from src.elasticity import ElasticityModel
from src.schema import EFFECT_COLS, TACTIC_COLS, Col, EffectTerm, pantry_columns
from src.utils import AppLogger

logger = AppLogger.get_logger(__name__)


class EffectCalculator:
    """Computes the price, promo, pantry and error effect terms (log scale).

    price = final promo elasticity x ln(price/base); promo = tactic contribution in
    promo weeks if positive; pantry = pl1..plN contribution; error = positive
    residual in promo weeks with a promo effect. Ineligible products get price only.
    """

    def __init__(
        self,
        models: Mapping[ModelName, ElasticityModel],
        priority: Sequence[ModelName],
        pantry_lags: int,
    ) -> None:
        self._models = dict(models)
        self._priority = list(priority)
        self._pantry = pantry_columns(pantry_lags)

    def compute(self, panel: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
        """Panel plus ln_pred, the four effect columns, base_raw and baseline_model."""
        assigned = self._assign_models(final)
        frame = panel.assign(**{Col.BASELINE_MODEL: panel[Col.UPC].map(assigned)})
        parts = self._model_terms(frame)
        delta = frame[Col.UPC].map(final.set_index(Col.UPC)[Col.PROMO_ELASTICITY])
        promo_week = frame[Col.PROMO_FLAG] == 1
        promo = parts["promo"].where(promo_week & (parts["promo"] > 0), 0.0)
        error = (frame[Col.LN_UNITS] - parts[Col.LN_PRED]).fillna(0.0)
        effects = {
            EFFECT_COLS[EffectTerm.PRICE]: (delta * frame[Col.LN_PROMO_PRICE]).fillna(0.0),
            EFFECT_COLS[EffectTerm.PROMO]: promo,
            EFFECT_COLS[EffectTerm.PANTRY]: parts["pantry"],
            EFFECT_COLS[EffectTerm.ERROR]: error.where(promo_week & (promo > 0) & (error > 0), 0.0),
        }
        out = frame.assign(**{Col.LN_PRED: parts[Col.LN_PRED]}, **effects)
        removed = out[list(effects)].sum(axis=1)
        return out.assign(**{Col.BASE_RAW: np.exp(out[Col.LN_UNITS] - removed)})

    def _assign_models(self, final: pd.DataFrame) -> pd.Series:
        """UPC -> model name: the selected model, else the first in priority (eligible only)."""
        fallback = self._priority[0].value
        chosen = final[Col.MODEL].where(final[Col.MODEL].notna(), fallback)
        chosen = chosen.where(final[Col.ELIGIBLE].astype(bool))
        return pd.Series(chosen.to_numpy(), index=final[Col.UPC])

    def _model_terms(self, frame: pd.DataFrame) -> pd.DataFrame:
        """ln_pred, promo and pantry contributions, computed per assigned model."""
        terms = pd.DataFrame({Col.LN_PRED: np.nan, "promo": 0.0, "pantry": 0.0}, index=frame.index)
        for name, rows in frame.groupby(Col.BASELINE_MODEL):
            model = self._models[ModelName(name)]
            terms.loc[rows.index, Col.LN_PRED] = model.predict(rows)
            terms.loc[rows.index, "promo"] = model.contribution(rows, TACTIC_COLS)
            terms.loc[rows.index, "pantry"] = model.contribution(rows, self._pantry)
        counts = frame[Col.BASELINE_MODEL].fillna("none").value_counts().to_dict()
        logger.info("Computed effect terms", extra={"rows_by_model": counts})
        return terms.fillna({"promo": 0.0, "pantry": 0.0})
