"""Promo options and their economics: every product x week x tactic x depth.

ExecutionLevels reads how widely each tactic is executed; EffectEstimator reads
tactic and pantry effects off the lift model at those levels; CandidateScorer turns
them into incremental volume, revenue, profit, trade spend and the objective's gain.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.elasticity import ElasticityModel
from src.schema import TACTIC_TERMS, Col, Tactic, pantry_columns

PRODUCT_INPUTS: tuple[Col, ...] = (
    Col.UPC, Col.REGULAR_PRICE, Col.UNIT_COST, Col.EFFECTIVE_COST, Col.PROMO_ELASTICITY,
)  # fmt: skip


@dataclass(frozen=True)
class PromoEffects:
    """Per-product log effects: one column per tactic (>= 0) and per pantry lag (<= 0).

    Both frames are indexed by UPC.
    """

    tactic: pd.DataFrame
    pantry: pd.DataFrame


class ExecutionLevels:
    """Typical store share of each tactic's columns in weeks the tactic ran.

    Tactic columns are shares of stores; a planned tactic is assumed to run as widely
    as it historically did (median share), not chain-wide, to avoid extrapolating.
    """

    def __init__(self, tactics: Sequence[Tactic], min_share: float) -> None:
        self._tactics = list(tactics)
        self._min_share = min_share

    def estimate(self, panel: pd.DataFrame) -> dict[Tactic, dict]:
        """Tactic -> {column: median share over weeks with every column >= min_share}."""
        levels = {}
        for tactic in self._tactics:
            cols = list(TACTIC_TERMS[tactic])
            ran = panel[(panel[cols] >= self._min_share).all(axis=1)]
            levels[tactic] = {c: float(ran[c].median()) if len(ran) else 1.0 for c in cols}
        return levels


class EffectEstimator:
    """Reads each product's tactic and pantry effects off the lift model.

    Effect = prediction with the tactic's columns at their execution level minus at 0.
    Tactic effects are floored at 0 and pantry effects capped at 0 (as in the baseline).
    """

    def __init__(self, model: ElasticityModel, pantry_lags: int) -> None:
        self._model = model
        self._pantry = pantry_columns(pantry_lags)

    def estimate(self, template: pd.DataFrame, levels: Mapping[Tactic, dict]) -> PromoEffects:
        """Effects from one no-promo row per product; unpredictable products get 0."""
        rows = template.drop_duplicates(Col.UPC).set_index(Col.UPC, drop=False)
        tactic = {t.value: self._effect(rows, lv).clip(lower=0) for t, lv in levels.items()}
        pantry = {c: self._effect(rows, {c: 1.0}).clip(upper=0) for c in self._pantry}
        return PromoEffects(pd.DataFrame(tactic).fillna(0.0), pd.DataFrame(pantry).fillna(0.0))

    def _effect(self, rows: pd.DataFrame, values: Mapping[str, float]) -> pd.Series:
        """Log effect of setting columns to `values` (from 0)."""
        return self._model.contribution(rows.assign(**values), list(values))


class CandidateScorer:
    """Scores every promo option over the horizon for one set of regular prices.

    Promo week units = baseline x (1 - d)^delta x exp(tactic effect); the dip removes
    baseline units in the next weeks (weeks past the horizon reuse the last week).
    Trade spend = d x regular price x promo units + fixed fee.
    """

    def __init__(
        self, tactics: Sequence[Tactic], depths: Sequence[float], fixed_cost: Mapping[Tactic, float]
    ) -> None:
        self._options = pd.DataFrame(
            [(t.value, d, fixed_cost[t]) for t in tactics for d in depths],
            columns=[Col.TACTIC, Col.DISCOUNT_DEPTH, Col.FIXED_COST],
        )

    def score(
        self, forecast: pd.DataFrame, products: pd.DataFrame, effects: PromoEffects
    ) -> pd.DataFrame:
        """One row per option with units, revenue, profit, spend, gain and ROI.

        `forecast` holds UPC, week, horizon week and forecast baseline; `products`
        holds PRODUCT_INPUTS. Gain charges the effective cost (and fees in proportion).
        """
        tactics = [t for t in self._options[Col.TACTIC].unique() if t in effects.tactic.columns]
        options = self._options[self._options[Col.TACTIC].isin(tactics)]
        weeks = forecast.assign(**{Col.DIP_UNITS: self._dip(forecast, effects.pantry)})
        frame = weeks.merge(products[list(PRODUCT_INPUTS)], on=Col.UPC).merge(options, how="cross")
        effect = effects.tactic.stack().rename("effect")
        frame = frame.join(effect, on=[Col.UPC, Col.TACTIC]).fillna({"effect": 0.0})
        return self._economics(frame).drop(columns="effect")

    @staticmethod
    def _dip(forecast: pd.DataFrame, pantry: pd.DataFrame) -> np.ndarray:
        """Units lost in the weeks after a promo in each row's week (pantry loading)."""
        ordered = forecast.sort_values([Col.UPC, Col.HORIZON_WEEK])
        by_upc = ordered.groupby(Col.UPC)[Col.FORECAST_BASELINE]
        last = by_upc.transform("last")
        dip = pd.Series(0.0, index=ordered.index)
        for lag, col in enumerate(pantry.columns, start=1):
            later = by_upc.shift(-lag).fillna(last)
            dip += later * (1 - np.exp(ordered[Col.UPC].map(pantry[col]).fillna(0.0)))
        return dip.reindex(forecast.index).to_numpy()

    @staticmethod
    def _economics(frame: pd.DataFrame) -> pd.DataFrame:
        """Adds lift, units, revenue, profit, spend, gain and ROI columns."""
        price, depth, base = (
            frame[Col.REGULAR_PRICE],
            frame[Col.DISCOUNT_DEPTH],
            frame[Col.FORECAST_BASELINE],
        )
        lift = np.exp(frame[Col.PROMO_ELASTICITY] * np.log1p(-depth) + frame["effect"])
        units = base * lift
        promo_price = price * (1 - depth)
        incremental = units - base - frame[Col.DIP_UNITS]
        revenue = promo_price * units - price * (base + frame[Col.DIP_UNITS])
        fee_weight = frame[Col.EFFECTIVE_COST] / frame[Col.UNIT_COST]
        spend = depth * price * units + frame[Col.FIXED_COST]
        gain = (
            revenue - frame[Col.EFFECTIVE_COST] * incremental - fee_weight * frame[Col.FIXED_COST]
        )
        return frame.assign(
            **{
                Col.PROMO_PRICE: promo_price,
                Col.LIFT: lift,
                Col.PROMO_UNITS: units,
                Col.INCREMENTAL: incremental,
                Col.INCREMENTAL_REVENUE: revenue,
                Col.INCREMENTAL_PROFIT: revenue
                - frame[Col.UNIT_COST] * incremental
                - frame[Col.FIXED_COST],
                Col.TRADE_SPEND: spend,
                Col.GAIN: gain,
                Col.ROI: gain / spend.where(spend > 0),
            }
        )
