"""Tests for src.baseline.decomposition.DueToDecomposer.

Lifts must always sum to incremental units; the split follows the log-effect
shares; the summary's baseline share is a known reference value.
One test class per production class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baseline.decomposition import DueToDecomposer
from src.schema import EFFECT_COLS, LIFT_COLS, BaselineFix, Col, EffectTerm


def _frame() -> pd.DataFrame:
    """Two weeks: a promo week (price 0.3, promo 0.1) and a corrected flat week."""
    effects = {
        EFFECT_COLS[EffectTerm.PRICE]: [0.3, 0.0],
        EFFECT_COLS[EffectTerm.PROMO]: [0.1, 0.0],
        EFFECT_COLS[EffectTerm.PANTRY]: [0.0, 0.0],
        EFFECT_COLS[EffectTerm.ERROR]: [0.0, 0.0],
    }
    return pd.DataFrame(
        {
            Col.UPC: 1, Col.DESCRIPTION: "A", Col.PRODUCT_SIZE: "1 OZ", Col.CATEGORY: "C",
            Col.UNITS: [140.0, 100.0], Col.BASELINE: [100.0, 90.0], Col.PROMO_FLAG: [1, 0],
            Col.BASELINE_FIX: [BaselineFix.NONE.value, BaselineFix.ROLLING.value], **effects,
        }
    )  # fmt: skip


class TestDueToDecomposer:
    """Incremental units are fully attributed."""

    def test_lifts_sum_to_incremental(self) -> None:
        """Promo week splits 40 units 3:1 between price and promo; flat week goes to error."""
        out = DueToDecomposer().decompose(_frame())
        lifts = out[list(LIFT_COLS.values())]
        np.testing.assert_allclose(lifts.sum(axis=1), out[Col.INCREMENTAL])
        assert out[LIFT_COLS[EffectTerm.PRICE]].iloc[0] == pytest.approx(30.0)
        assert out[LIFT_COLS[EffectTerm.PROMO]].iloc[0] == pytest.approx(10.0)
        assert out[LIFT_COLS[EffectTerm.ERROR]].iloc[1] == pytest.approx(10.0)

    def test_summary(self) -> None:
        """Baseline share = 190 / 240; promo-week lift = 140/100 - 1."""
        decomposer = DueToDecomposer()
        summary = decomposer.summarize(decomposer.decompose(_frame())).iloc[0]
        assert summary[Col.BASELINE_SHARE] == pytest.approx(190 / 240)
        assert summary[Col.PROMO_WEEK_LIFT] == pytest.approx(0.4)
        assert summary[f"weeks_{BaselineFix.ROLLING.value}"] == 1
