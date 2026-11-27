"""Due-to decomposition: splits incremental units across the effect terms.

Incremental = actual - baseline. It is allocated to price, promo, pantry and error
in proportion to their log effects, so the lifts always sum to the incremental
units. Weeks whose baseline was corrected without effects carry it as error.
"""

from __future__ import annotations

import pandas as pd

from src.schema import EFFECT_COLS, LIFT_COLS, BaselineFix, Col, EffectTerm

EPSILON = 1e-9


class DueToDecomposer:
    """Adds incremental_units and one lift column per effect term (units scale).

    Also summarises each product: baseline share of volume and average promo-week lift.
    """

    def decompose(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Frame plus incremental units and per-term lifts that sum to it."""
        incremental = frame[Col.UNITS] - frame[Col.BASELINE]
        effects = frame[list(EFFECT_COLS.values())]
        total = effects.sum(axis=1)
        has_effects = total.abs() > EPSILON
        shares = effects.div(total.where(has_effects), axis=0).fillna(0.0)
        lifts = shares.mul(incremental, axis=0)
        lifts.columns = [LIFT_COLS[t] for t in EFFECT_COLS]
        error_lift = LIFT_COLS[EffectTerm.ERROR]
        lifts[error_lift] = lifts[error_lift].where(has_effects, incremental)
        return frame.assign(**{Col.INCREMENTAL: incremental}, **lifts)

    def summarize(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per product: units, baseline, baseline share, promo-week lift, fix counts."""
        promo = frame[frame[Col.PROMO_FLAG] == 1]
        lift = (promo[Col.UNITS] / promo[Col.BASELINE] - 1).groupby(promo[Col.UPC]).mean()
        grouped = frame.groupby([Col.UPC, Col.DESCRIPTION, Col.PRODUCT_SIZE, Col.CATEGORY])
        summary = grouped.agg(
            **{
                Col.UNITS: (Col.UNITS, "sum"),
                Col.BASELINE: (Col.BASELINE, "sum"),
                **{c: (c, "sum") for c in LIFT_COLS.values()},
            }
        ).reset_index()
        fixes = pd.crosstab(frame[Col.UPC], frame[Col.BASELINE_FIX]).reindex(
            columns=[f.value for f in BaselineFix], fill_value=0
        )
        fixes.columns = [f"weeks_{c}" for c in fixes.columns]
        return summary.assign(
            **{
                Col.BASELINE_SHARE: summary[Col.BASELINE] / summary[Col.UNITS],
                Col.PROMO_WEEK_LIFT: summary[Col.UPC].map(lift),
            }
        ).merge(fixes, left_on=Col.UPC, right_index=True, how="left")
