"""Facade and composition root for the data-prep layer.

DataPrepService runs load -> panel -> features -> trend and writes the panel;
DataPrepFactory wires its collaborators from config and registries. Other
layers import only these two (via `from src.prep import ...`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config
from src.prep.features import FEATURE_STEPS, FeatureBuilder
from src.prep.loader import LoaderSettings, RawDataLoader
from src.prep.panel import (
    AGGREGATORS,
    PRODUCT_FILTERS,
    PanelBuilder,
    ProductEligibility,
    TransactionCleaner,
)
from src.prep.trend import LowessTrend
from src.schema import Col, Split, driver_columns
from src.utils import AppLogger

logger = AppLogger.get_logger(__name__)


class DataPrepService:
    """Facade for Phase 1: raw workbook -> featured, trended panel parquet.

    All collaborators are injected; build one with DataPrepFactory.
    """

    def __init__(
        self,
        loader: RawDataLoader,
        panel_builder: PanelBuilder,
        feature_builder: FeatureBuilder,
        trend: LowessTrend,
        output_path: Path,
    ) -> None:
        self._loader = loader
        self._panel_builder = panel_builder
        self._feature_builder = feature_builder
        self._trend = trend
        self._output_path = output_path

    def run(self, use_cache: bool = True) -> pd.DataFrame:
        """Build the panel, write it to output_path and return it."""
        panel = self._panel_builder.build(self._loader.load(use_cache=use_cache))
        panel = self._trend.apply(self._feature_builder.build(panel))
        panel = panel.sort_values(list(self._panel_builder.keys)).reset_index(drop=True)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(self._output_path, index=False)
        logger.info(
            "Wrote panel",
            extra={
                "path": self._output_path.name,
                "rows": len(panel),
                "products": panel[Col.UPC].nunique(),
                "weeks": panel[Col.WEEK].nunique(),
                "test_rows": int((panel[Col.SPLIT] == Split.TEST).sum()),
            },
        )
        logger.print("Panel columns", list(panel.columns))
        return panel


class DataPrepFactory:
    """Builds a DataPrepService from config plus loader settings and output path.

    The only place that instantiates prep classes, so tests can swap paths.
    """

    def __init__(self, cfg: Config, loader_settings: LoaderSettings, output_path: Path) -> None:
        self._cfg = cfg
        self._loader_settings = loader_settings
        self._output_path = output_path

    def build(self) -> DataPrepService:
        """Wire registered aggregator, filters and feature steps into the service."""
        cfg = self._cfg
        eligibility = ProductEligibility([f.from_config(cfg.filters) for f in PRODUCT_FILTERS])
        panel_builder = PanelBuilder(
            TransactionCleaner(), AGGREGATORS.get(cfg.data.grain)(), eligibility
        )
        return DataPrepService(
            loader=RawDataLoader(self._loader_settings),
            panel_builder=panel_builder,
            feature_builder=FeatureBuilder([s.from_config(cfg) for s in FEATURE_STEPS]),
            trend=LowessTrend(
                cfg.trend.lowess_window_weeks,
                drivers=driver_columns(cfg.features.fourier_order, cfg.features.pantry_lags),
            ),
            output_path=self._output_path,
        )
