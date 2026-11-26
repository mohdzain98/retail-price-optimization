"""Frequentist elasticity models: per-product OLS, pooled OLS and MixedLM.

Each estimates a regular-price and a promo-price slope. Per-product OLS is
unbiased but noisy, pooled OLS forces one slope per category, and MixedLM
partially pools product slopes toward the category (RGMx-style mixed model).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.config import ElasticityConfig, ModelName
from src.elasticity.base import (
    ELASTICITY_MODELS,
    SLOPE_COLS,
    DesignSpec,
    ElasticityModel,
    PriceCentering,
)
from src.schema import ESTIMATE_COLS, Col, Family
from src.utils import AppLogger, DataValidationError

logger = AppLogger.get_logger(__name__)

CONST = "const"
RE_CONST = "re_const"


def _estimate_rows(
    upcs: list,
    names: dict[Family, list[str]],
    params: pd.Series,
    ci: pd.DataFrame,
    shift: dict[Family, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Estimate frame from coefficient names per family (missing names give NaN)."""
    out: dict[str, object] = {Col.UPC: upcs}
    for family, (est, low, high) in ESTIMATE_COLS.items():
        offset = shift[family] if shift else 0.0
        out[est] = params.reindex(names[family]).to_numpy() + offset
        out[low] = ci.reindex(names[family]).iloc[:, 0].to_numpy() + offset
        out[high] = ci.reindex(names[family]).iloc[:, 1].to_numpy() + offset
    return pd.DataFrame(out)


class _SlopeDesign:
    """Per-category slopes for both price terms plus active controls.

    Shared by pooled OLS and MixedLM so both use the same fixed-effect layout.
    """

    def __init__(self, spec: DesignSpec, centering: PriceCentering) -> None:
        self._spec = spec
        self._centering = centering
        self._categories: list[str] = []
        self._controls: list[str] = []

    @staticmethod
    def slope_name(family: Family, category: str) -> str:
        """Column name of a category's slope for one price term."""
        return f"{family.value}_slope[{category}]"

    @property
    def categories(self) -> list[str]:
        """Categories seen in training, sorted."""
        return self._categories

    def fit(self, train: pd.DataFrame) -> None:
        """Remember categories and controls that vary in training."""
        self._categories = sorted(train[Col.CATEGORY].unique())
        self._controls = self._spec.active_controls(train)

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Slope columns (price term x category indicator) and controls."""
        slopes = self._centering.transform(frame)
        columns = {
            self.slope_name(family, cat): slopes[col] * (frame[Col.CATEGORY] == cat)
            for family, col in SLOPE_COLS.items()
            for cat in self._categories
        }
        design = pd.DataFrame(columns, index=frame.index)
        return pd.concat([design, frame[self._controls]], axis=1).astype(float)

    def names_for(self, train: pd.DataFrame, upcs: list) -> dict[Family, list[str]]:
        """Each product's category slope names, per family."""
        category = train.drop_duplicates(Col.UPC).set_index(Col.UPC)[Col.CATEGORY]
        return {f: [self.slope_name(f, category[u]) for u in upcs] for f in SLOPE_COLS}


@ELASTICITY_MODELS.register
class OlsPerProduct(ElasticityModel):
    """Separate OLS per product: own intercept, both slopes and control effects.

    A slope whose regressor never varies for a product is left out (NaN estimate).
    """

    name = ModelName.OLS_PER_PRODUCT

    def __init__(self, spec: DesignSpec, settings: ElasticityConfig) -> None:
        super().__init__(spec, settings)
        self._params: dict[int, pd.Series] = {}

    def fit(self, train: pd.DataFrame) -> pd.DataFrame:
        """Fit one regression per UPC on its training rows."""
        self._centering.fit(train)
        rows = [self._fit_product(upc, group) for upc, group in train.groupby(Col.UPC)]
        return pd.concat(rows, ignore_index=True)

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Per-product linear prediction; NaN for unfitted products."""
        out = pd.Series(np.nan, index=frame.index)
        for upc, group in frame.groupby(Col.UPC):
            if upc in self._params:
                params = self._params[upc]
                out.loc[group.index] = self._design(group)[params.index] @ params
        return out

    def _fit_product(self, upc: int, group: pd.DataFrame) -> pd.DataFrame:
        """OLS for one product on regressors that vary; returns its estimate row."""
        design = self._design(group)
        allowed = {CONST, *SLOPE_COLS.values(), *self._spec.active_controls(group)}
        keep = [c for c in design.columns if c in allowed and (c == CONST or design[c].std() > 0)]
        result = sm.OLS(group[self._spec.response], design[keep]).fit()
        self._params[upc] = result.params
        names = {family: [col] for family, col in SLOPE_COLS.items()}
        return _estimate_rows([upc], names, result.params, result.conf_int(self.interval_alpha))

    def _design(self, frame: pd.DataFrame) -> pd.DataFrame:
        """[const, slope regressors, controls present in frame]."""
        controls = [c for c in self._spec.controls if c in frame.columns]
        const = pd.DataFrame({CONST: 1.0}, index=frame.index)
        slopes = self._centering.transform(frame)
        return pd.concat([const, slopes, frame[controls]], axis=1).astype(float)


@ELASTICITY_MODELS.register
class PooledOls(ElasticityModel):
    """One OLS with product intercepts and one slope per category and price term.

    Stable, but every product in a category gets the same elasticities.
    """

    name = ModelName.POOLED_OLS

    def __init__(self, spec: DesignSpec, settings: ElasticityConfig) -> None:
        super().__init__(spec, settings)
        self._slopes = _SlopeDesign(spec, self._centering)
        self._upcs: list[int] = []
        self._params: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> pd.DataFrame:
        """Fit the pooled regression; each UPC takes its category slopes."""
        self._centering.fit(train)
        self._slopes.fit(train)
        self._upcs = sorted(train[Col.UPC].unique())
        result = sm.OLS(train[self._spec.response], self._design(train)).fit()
        self._params = result.params
        names = self._slopes.names_for(train, self._upcs)
        ci = result.conf_int(self.interval_alpha)
        return _estimate_rows(self._upcs, names, result.params, ci)

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Linear prediction; NaN for products not in training."""
        if self._params is None:
            raise DataValidationError("PooledOls.predict called before fit")
        pred = self._design(frame) @ self._params
        return pred.where(frame[Col.UPC].isin(self._upcs))

    def _design(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Product intercept dummies + category slopes + controls."""
        dummies = {f"upc[{u}]": (frame[Col.UPC] == u).astype(float) for u in self._upcs}
        intercepts = pd.DataFrame(dummies, index=frame.index)
        return pd.concat([intercepts, self._slopes.build(frame)], axis=1)


@ELASTICITY_MODELS.register
class MixedEffects(ElasticityModel):
    """MixedLM: category fixed slopes plus random intercept and slopes per product.

    Product slopes shrink toward the category slope (partial pooling); intervals use
    the fixed-slope standard error shifted by the product's random slope.
    """

    name = ModelName.MIXED_EFFECTS

    def __init__(self, spec: DesignSpec, settings: ElasticityConfig) -> None:
        super().__init__(spec, settings)
        self._slopes = _SlopeDesign(spec, self._centering)
        self._fe: pd.Series | None = None
        self._re = pd.DataFrame()

    def fit(self, train: pd.DataFrame) -> pd.DataFrame:
        """Fit MixedLM by REML; logs (does not hide) convergence warnings."""
        self._centering.fit(train)
        self._slopes.fit(train)
        model = MixedLM(
            train[self._spec.response], self._fixed(train),
            groups=train[Col.UPC], exog_re=self._random(train),
        )  # fmt: skip
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            result = model.fit(reml=True, method=["lbfgs"])
        for warning in caught:
            logger.warning("MixedLM: %s", warning.message)
        self._fe = result.fe_params
        self._re = pd.DataFrame(result.random_effects).T
        upcs = list(self._re.index)
        shift = {f: self._re[col].to_numpy() for f, col in SLOPE_COLS.items()}
        names = self._slopes.names_for(train, upcs)
        ci = result.conf_int(self.interval_alpha)
        return _estimate_rows(upcs, names, result.fe_params, ci, shift)

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Fixed part + product random intercept and slopes; NaN for unseen products."""
        if self._fe is None:
            raise DataValidationError("MixedEffects.predict called before fit")
        effects = self._re.reindex(frame[Col.UPC].to_numpy())
        effects.index = frame.index
        random_design = self._random(frame)
        random_part = (random_design * effects[random_design.columns]).sum(axis=1, min_count=1)
        return self._fixed(frame) @ self._fe + random_part

    def _fixed(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Constant + category intercepts (first dropped) + category slopes + controls."""
        cats = self._slopes.categories
        intercepts = {f"cat[{c}]": (frame[Col.CATEGORY] == c).astype(float) for c in cats[1:]}
        base = pd.DataFrame({CONST: 1.0, **intercepts}, index=frame.index)
        return pd.concat([base, self._slopes.build(frame)], axis=1)

    def _random(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Random-effects design: intercept and both slope regressors."""
        const = pd.DataFrame({RE_CONST: 1.0}, index=frame.index)
        return pd.concat([const, self._centering.transform(frame)], axis=1)
