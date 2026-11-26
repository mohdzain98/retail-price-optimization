"""Facade and composition root for the elasticity layer (Phase 2).

ElasticityService fits every configured model on eligible training rows, scores
them on the holdout and on per-product quality gates, selects a model per product
by priority, imputes the rest hierarchically and writes outputs.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ELASTICITY_ALL_MODELS_PATH,
    ELASTICITY_PATH,
    MODEL_COMPARISON_PATH,
    Config,
    ModelName,
)
from src.elasticity.base import ELASTICITY_MODELS, DesignSpec, ElasticityModel, HoldoutScorer
from src.elasticity.guardrails import (
    PRODUCT_COLS,
    HierarchicalImputer,
    ModelSelector,
    QualityGates,
)
from src.schema import Col, Family, Split
from src.utils import AppLogger

logger = AppLogger.get_logger(__name__)


@dataclass(frozen=True)
class ElasticityOutputs:
    """Where the stage writes its tables and model artifacts."""

    final: Path
    all_models: Path
    comparison: Path
    artifacts_dir: Path

    @classmethod
    def default(cls) -> ElasticityOutputs:
        """Standard locations under data/processed."""
        return cls(
            ELASTICITY_PATH,
            ELASTICITY_ALL_MODELS_PATH,
            MODEL_COMPARISON_PATH,
            ELASTICITY_PATH.parent,
        )


@dataclass(frozen=True)
class ElasticityReport:
    """Results of one run: final table, every model's scored estimates, comparison."""

    final: pd.DataFrame
    all_models: pd.DataFrame
    comparison: pd.DataFrame


@dataclass(frozen=True)
class PostProcessing:
    """The three post-processing steps applied after fitting."""

    gates: QualityGates
    selector: ModelSelector
    imputer: HierarchicalImputer


class ElasticityService:
    """Facade: fit models, score them, select per product, impute, write outputs.

    Models only see eligible products; the imputer covers the rest.
    """

    def __init__(
        self,
        models: Sequence[ElasticityModel],
        post: PostProcessing,
        scorer: HoldoutScorer,
        outputs: ElasticityOutputs,
    ) -> None:
        self._models = list(models)
        self._post = post
        self._scorer = scorer
        self._outputs = outputs

    def run(self, panel: pd.DataFrame) -> ElasticityReport:
        """Fit, score, select, impute and persist; returns the report."""
        eligible = panel[panel[Col.ELIGIBLE].astype(bool)]
        train = eligible[eligible[Col.SPLIT] == Split.TRAIN]
        test = eligible[eligible[Col.SPLIT] == Split.TEST]
        scored, rows = {}, []
        for model in self._models:
            scored[model.name], row = self._fit_and_score(model, train, test)
            rows.append(row)
        products = panel.drop_duplicates(Col.UPC)[list(PRODUCT_COLS)]
        final = self._post.imputer.apply(self._post.selector.select(scored), products)
        report = ElasticityReport(final, self._long(scored, products), pd.DataFrame(rows))
        self._write(report)
        return report

    def _fit_and_score(
        self, model: ElasticityModel, train: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        """Fit one model, time it, score holdout and gates, summarise."""
        logger.info("Fitting model", extra={"model": model.name.value})
        start = time.perf_counter()
        estimates = model.fit(train)
        seconds = time.perf_counter() - start
        scored = self._post.gates.evaluate(model, estimates, train)
        low, high = self._post.gates.bounds
        beta = scored[Col.ELASTICITY]
        row = {
            Col.MODEL: model.name.value,
            Col.WAPE: self._scorer.score(model, test).wape,
            Col.N_PRODUCTS: len(scored),
            Col.FIT_SECONDS: round(seconds, 1),
            Col.MEDIAN_ELASTICITY: float(beta.median()),
            Col.SHARE_IN_BOUNDS: float(beta.between(low, high).mean()),
            Col.SHARE_PASSING: float(scored[Col.PASSES_GATES].mean()),
        }
        logger.info("Model done", extra={k: _rounded(v) for k, v in row.items()})
        model.save_artifacts(self._outputs.artifacts_dir)
        return scored, row

    @staticmethod
    def _long(scored: dict[ModelName, pd.DataFrame], products: pd.DataFrame) -> pd.DataFrame:
        """All models' scored estimates stacked with product attributes."""
        frames = [est.assign(**{Col.MODEL: name.value}) for name, est in scored.items()]
        stacked = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return products.merge(stacked, on=Col.UPC, how="inner")

    def _write(self, report: ElasticityReport) -> None:
        """Persist the three tables as parquet."""
        for frame, path in (
            (report.final, self._outputs.final),
            (report.all_models, self._outputs.all_models),
            (report.comparison, self._outputs.comparison),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.replace({np.inf: np.nan}).to_parquet(path, index=False)
        logger.info("Wrote elasticity outputs", extra={"final": self._outputs.final.name})


def _rounded(value: object) -> object:
    """Round floats for log readability."""
    return round(value, 3) if isinstance(value, float) else value


class ElasticityFactory:
    """Builds an ElasticityService from config (models come from the registry).

    The only place that instantiates elasticity classes.
    """

    def __init__(self, cfg: Config, outputs: ElasticityOutputs) -> None:
        self._cfg = cfg
        self._outputs = outputs

    def build(self) -> ElasticityService:
        """Instantiate configured models in order and wire the post-processing."""
        settings = self._cfg.elasticity
        spec = DesignSpec.from_config(self._cfg)
        models = [ELASTICITY_MODELS.get(name)(spec, settings) for name in settings.models]
        bounds = {Family.REGULAR: settings.bounds, Family.PROMO: settings.promo_bounds}
        post = PostProcessing(
            gates=QualityGates(settings.quality, settings.bounds),
            selector=ModelSelector(settings.model_priority),
            imputer=HierarchicalImputer(settings.imputation_levels, bounds),
        )
        return ElasticityService(models, post, HoldoutScorer(), self._outputs)
