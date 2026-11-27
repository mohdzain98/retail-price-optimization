"""Bayesian hierarchical elasticity model (PyMC).

Regular-price and promo-price slopes each follow global -> category -> product
hierarchies (non-centred partial pooling); control effects vary by category.
Student-t noise resists launch/phase-out outliers; draws are kept for Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.config import ElasticityConfig, ModelName
from src.elasticity.base import ELASTICITY_MODELS, SLOPE_COLS, DesignSpec, ElasticityModel
from src.schema import ESTIMATE_COLS, Col, Family
from src.utils import AppLogger, DataValidationError

logger = AppLogger.get_logger(__name__)

DRAW = "draw"


@dataclass(frozen=True)
class _ModelData:
    """Arrays fed to PyMC plus the coordinate labels."""

    y: np.ndarray
    slopes: np.ndarray  # rows x families, in SLOPE_COLS order
    controls: np.ndarray
    upc_idx: np.ndarray
    cat_of_upc: np.ndarray
    upcs: list[int]
    categories: list[str]
    control_names: list[str]
    y_mean: np.ndarray


@dataclass(frozen=True)
class _PosteriorMeans:
    """Posterior means used for point predictions (gamma: category x control)."""

    alpha: pd.Series
    beta: pd.DataFrame  # upc x family
    gamma: pd.DataFrame


class _ControlScaler:
    """Demeans controls per product, then scales by the global train std.

    Per-product demeaning keeps control effects separate from product levels (alpha).
    """

    def __init__(self) -> None:
        self._means = pd.DataFrame()
        self._std = pd.Series(dtype=float)

    def fit(self, train: pd.DataFrame, names: list[str]) -> None:
        """Learn per-product means and global std of the within-product deviations."""
        self._means = train.groupby(Col.UPC)[names].mean()
        deviations = train[names] - self._means.reindex(train[Col.UPC]).to_numpy()
        self._std = deviations.std().replace(0, 1.0)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Scaled within-product control deviations as a float matrix."""
        names = list(self._std.index)
        means = self._means.reindex(frame[Col.UPC]).to_numpy()
        return ((frame[names].to_numpy() - means) / self._std.to_numpy()).astype(float)


@ELASTICITY_MODELS.register
class BayesianHierarchical(ElasticityModel):
    """Hierarchical log-log model with regular and promo price slopes per product.

    Reports posterior mean and HDI per product and family; logs r-hat, ESS, divergences.
    """

    name = ModelName.BAYES_HIERARCHICAL
    ALPHA_SD: ClassVar[float] = 1.0  # product level around its mean log units
    GAMMA_SD: ClassVar[float] = 0.5  # standardized control effects
    NOISE_SD: ClassVar[float] = 0.5
    NU_ALPHA, NU_BETA = 2.0, 0.1  # Student-t degrees of freedom prior
    MAX_RHAT: ClassVar[float] = 1.01
    FAMILIES: ClassVar[tuple[Family, ...]] = tuple(SLOPE_COLS)

    def __init__(self, spec: DesignSpec, settings: ElasticityConfig) -> None:
        super().__init__(spec, settings)
        self._scaler = _ControlScaler()
        self._means: _PosteriorMeans | None = None
        self._draws = pd.DataFrame()

    def fit(self, train: pd.DataFrame) -> pd.DataFrame:
        """Sample the posterior and return mean + HDI per product for both families."""
        self._centering.fit(train)
        data = self._prepare(train)
        cfg = self._settings.bayes
        with self._build(data):
            idata = pm.sample(
                draws=cfg.draws, tune=cfg.tune, chains=cfg.chains, cores=cfg.chains,
                target_accept=cfg.target_accept, random_seed=cfg.seed, progressbar=False,
                mp_ctx="spawn",  # fork() from a threaded process can deadlock
            )  # fmt: skip
        self._log_diagnostics(idata)
        self._store_posterior(idata, data)
        return self._summarize(data.upcs)

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Posterior-mean linear predictor; NaN for products not in training."""
        if self._means is None:
            raise DataValidationError("BayesianHierarchical.predict called before fit")
        slopes = self._centering.transform(frame)
        betas = self._means.beta.reindex(frame[Col.UPC]).to_numpy()
        price_part = (slopes[list(SLOPE_COLS.values())].to_numpy() * betas).sum(axis=1)
        alpha = frame[Col.UPC].map(self._means.alpha)
        return (alpha + price_part + self._control_effect(frame)).astype(float)

    def __getstate__(self) -> dict:
        """Pickle without the (large) draws; they are saved separately as parquet."""
        return {**self.__dict__, "_draws": pd.DataFrame()}

    def save_artifacts(self, directory: Path) -> None:
        """Write posterior elasticity draws (long format) for robustness analysis."""
        directory.mkdir(parents=True, exist_ok=True)
        self._draws.to_parquet(directory / "elasticity_draws.parquet", index=False)

    def _control_effect(self, frame: pd.DataFrame) -> pd.Series:
        """Row-wise sum of scaled controls times their category's effects."""
        if self._means is None:
            raise DataValidationError("posterior not stored")
        effects = self._means.gamma.reindex(frame[Col.CATEGORY]).to_numpy()
        return pd.Series((self._scaler.transform(frame) * effects).sum(axis=1), index=frame.index)

    def _prepare(self, train: pd.DataFrame) -> _ModelData:
        """Index products/categories and build the standardized design arrays."""
        controls = self._spec.active_controls(train)
        self._scaler.fit(train, controls)
        upcs = sorted(train[Col.UPC].unique())
        categories = sorted(train[Col.CATEGORY].unique())
        cat_of = train.drop_duplicates(Col.UPC).set_index(Col.UPC).loc[upcs, Col.CATEGORY]
        y = train[self._spec.response].to_numpy(float)
        upc_idx = pd.Categorical(train[Col.UPC], categories=upcs).codes
        slopes = self._centering.transform(train)[list(SLOPE_COLS.values())].to_numpy(float)
        return _ModelData(
            y=y, slopes=slopes, controls=self._scaler.transform(train), upc_idx=upc_idx,
            cat_of_upc=pd.Categorical(cat_of, categories=categories).codes,
            upcs=upcs, categories=categories, control_names=controls,
            y_mean=pd.Series(y).groupby(upc_idx).mean().to_numpy(),
        )  # fmt: skip

    def _priors(self, family: Family) -> tuple[float, float]:
        """(mean, sd) of the global slope prior for a family."""
        cfg = self._settings.bayes
        if family is Family.PROMO:
            return cfg.promo_prior_mean, cfg.promo_prior_sd
        return cfg.prior_mean, cfg.prior_sd

    def _slope(self, family: Family, data: _ModelData) -> Any:
        """Non-centred global -> category -> product slope for one family."""
        mean, sd = self._priors(family)
        group_sd = self._settings.bayes.prior_group_sd
        f = family.value
        mu_global = pm.Normal(f"mu_global_{f}", mean, sd)
        z_cat = pm.Normal(f"z_cat_{f}", 0, 1, dims="category")
        mu_cat = pm.Deterministic(
            f"mu_cat_{f}", mu_global + pm.HalfNormal(f"sigma_cat_{f}", group_sd) * z_cat,
            dims="category",
        )  # fmt: skip
        z_prod = pm.Normal(f"z_prod_{f}", 0, 1, dims="upc")
        sigma_prod = pm.HalfNormal(f"sigma_prod_{f}", group_sd)
        return pm.Deterministic(
            f"beta_{f}", mu_cat[data.cat_of_upc] + sigma_prod * z_prod, dims="upc"
        )

    def _build(self, data: _ModelData) -> pm.Model:
        """Two slope hierarchies, product intercepts, category control effects, Student-t."""
        coords = {"upc": data.upcs, "category": data.categories, "control": data.control_names}
        with pm.Model(coords=coords) as model:
            betas = [self._slope(family, data) for family in self.FAMILIES]
            alpha = pm.Normal("alpha", data.y_mean, self.ALPHA_SD, dims="upc")
            gamma = pm.Normal("gamma", 0, self.GAMMA_SD, dims=("category", "control"))
            row_cat = data.cat_of_upc[data.upc_idx]
            price = sum(b[data.upc_idx] * data.slopes[:, i] for i, b in enumerate(betas))
            mu = alpha[data.upc_idx] + price + (data.controls * gamma[row_cat]).sum(axis=1)
            pm.StudentT(
                "y", nu=pm.Gamma("nu", self.NU_ALPHA, self.NU_BETA),
                mu=mu, sigma=pm.HalfNormal("sigma", self.NOISE_SD), observed=data.y,
            )  # fmt: skip
        return model

    def _log_diagnostics(self, idata: Any) -> None:
        """Log max r-hat, min ESS and divergences; warn if the sampler struggled."""
        names = [f"{p}_{f.value}" for f in self.FAMILIES for p in ("beta", "mu_cat")]
        summary = az.summary(idata, var_names=names)
        r_hat = float(pd.to_numeric(summary["r_hat"]).max())
        ess = float(pd.to_numeric(summary["ess_bulk"]).min())
        divergences = int(idata.sample_stats["diverging"].sum())
        extra = {"max_r_hat": round(r_hat, 3), "min_ess_bulk": int(ess), "divergences": divergences}
        if r_hat > self.MAX_RHAT or divergences > 0:
            logger.warning("Sampler diagnostics need attention", extra=extra)
        else:
            logger.info("Sampler diagnostics OK", extra=extra)

    def _store_posterior(self, idata: Any, data: _ModelData) -> None:
        """Keep posterior means for prediction and flattened elasticity draws."""
        post = idata.posterior
        n = len(data.upcs)
        samples = {f: post[f"beta_{f.value}"].to_numpy().reshape(-1, n) for f in self.FAMILIES}
        n_draws = next(iter(samples.values())).shape[0]
        index = pd.MultiIndex.from_product([range(n_draws), data.upcs], names=[DRAW, Col.UPC])
        flat = {ESTIMATE_COLS[f][0]: s.reshape(-1) for f, s in samples.items()}  # draw-major
        self._draws = pd.DataFrame(flat, index=index).reset_index()
        mean = post.mean(("chain", "draw"))
        self._means = _PosteriorMeans(
            alpha=pd.Series(mean["alpha"].to_numpy(), index=data.upcs),
            beta=pd.DataFrame({f: s.mean(axis=0) for f, s in samples.items()}, index=data.upcs),
            gamma=pd.DataFrame(
                mean["gamma"].to_numpy(), index=data.categories, columns=data.control_names
            ),
        )

    def _summarize(self, upcs: list[int]) -> pd.DataFrame:
        """Posterior mean and HDI per product and family from the stored draws."""
        if self._means is None:
            raise DataValidationError("posterior not stored")
        prob = self._settings.bayes.hdi_prob
        out: dict[str, object] = {Col.UPC: upcs}
        for family in self.FAMILIES:
            est, low, high = ESTIMATE_COLS[family]
            grouped = self._draws.groupby(Col.UPC)[est]
            intervals = {upc: _hdi(group.to_numpy(), prob) for upc, group in grouped}
            out[est] = self._means.beta.loc[upcs, family].to_numpy()
            out[low] = [intervals[u][0] for u in upcs]
            out[high] = [intervals[u][1] for u in upcs]
        return pd.DataFrame(out)


def _hdi(samples: np.ndarray, prob: float) -> tuple[float, float]:
    """Narrowest interval containing `prob` of the samples (version-independent HDI)."""
    ordered = np.sort(samples)
    width = int(np.ceil(prob * len(ordered)))
    spans = ordered[width - 1 :] - ordered[: len(ordered) - width + 1]
    start = int(np.argmin(spans))
    return float(ordered[start]), float(ordered[start + width - 1])
