"""Facade and composition root for the baseline layer (Phase 3).

BaselineService runs effects -> bad-baseline correction -> due-to decomposition and
writes the weekly table and the per-product summary; BaselineFactory wires it from
config, registered rules and the fitted elasticity models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.baseline.decomposition import DueToDecomposer
from src.baseline.effects import EffectCalculator
from src.baseline.rules import BASELINE_RULES, BaselineCorrector
from src.config import BASELINE_PATH, BASELINE_SUMMARY_PATH, Config, ModelName
from src.elasticity import ElasticityModel
from src.schema import Col
from src.utils import AppLogger

logger = AppLogger.get_logger(__name__)


@dataclass(frozen=True)
class BaselineOutputs:
    """Where the stage writes the weekly baseline and the product summary."""

    weekly: Path
    summary: Path

    @classmethod
    def default(cls) -> BaselineOutputs:
        """Standard locations under data/processed."""
        return cls(BASELINE_PATH, BASELINE_SUMMARY_PATH)


@dataclass(frozen=True)
class BaselineReport:
    """Weekly decomposition and per-product summary."""

    weekly: pd.DataFrame
    summary: pd.DataFrame


class BaselineService:
    """Facade: effect terms, RGMx bad-baseline fixes, due-to lifts, outputs.

    Inputs are the prepared panel and the final elasticity table.
    """

    def __init__(
        self,
        calculator: EffectCalculator,
        corrector: BaselineCorrector,
        decomposer: DueToDecomposer,
        outputs: BaselineOutputs,
    ) -> None:
        self._calculator = calculator
        self._corrector = corrector
        self._decomposer = decomposer
        self._outputs = outputs

    def run(self, panel: pd.DataFrame, final: pd.DataFrame) -> BaselineReport:
        """Build, correct and decompose the baseline; write and return the report."""
        weekly = self._calculator.compute(panel, final)
        weekly = self._decomposer.decompose(self._corrector.correct(weekly))
        report = BaselineReport(weekly, self._decomposer.summarize(weekly))
        for frame, path in (
            (report.weekly, self._outputs.weekly),
            (report.summary, self._outputs.summary),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
        share = report.weekly[Col.BASELINE].sum() / report.weekly[Col.UNITS].sum()
        logger.info(
            "Wrote baseline", extra={"rows": len(weekly), "baseline_share": round(share, 3)}
        )
        return report


class BaselineFactory:
    """Builds a BaselineService from config and the fitted elasticity models.

    The only place that instantiates baseline classes.
    """

    def __init__(
        self, cfg: Config, models: Mapping[ModelName, ElasticityModel], outputs: BaselineOutputs
    ) -> None:
        self._cfg = cfg
        self._models = models
        self._outputs = outputs

    def build(self) -> BaselineService:
        """Wire the effect calculator, registered rules and decomposer."""
        cfg = self._cfg
        calculator = EffectCalculator(
            self._models, cfg.elasticity.model_priority, cfg.features.pantry_lags
        )
        corrector = BaselineCorrector(
            [rule.from_config(cfg.baseline) for rule in BASELINE_RULES],
            cfg.baseline.rolling_window,
            cfg.baseline.promo_lookback_weeks,
        )
        return BaselineService(calculator, corrector, DueToDecomposer(), self._outputs)
