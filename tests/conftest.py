"""Shared fixtures: a synthetic Breakfast at the Frat dataset with known elasticity.

SyntheticFrat generates store x UPC x week transactions in the raw schema
(upper-case columns, title row above the header) so loading, cleaning and
features are tested without the licensed data. True elasticities are exposed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from src.config import Config, Sheet
from src.prep import DataPrepFactory, LoaderSettings, RawData, RawDataLoader


@dataclass(frozen=True)
class SyntheticProduct:
    """One synthetic UPC: category, true elasticity, base price, price changes."""

    category: str
    elasticity: float
    base_price: float
    varies: bool = True


class SyntheticFrat:
    """Generates raw sheets with ln Q = a + beta ln(P/base) + promo + season + noise.

    FLAT_UPC never changes price, so it must fail the price-variation filter.
    """

    N_WEEKS: ClassVar[int] = 104
    STORES: ClassVar[tuple[int, ...]] = (101, 102, 103)
    FLAT_UPC: ClassVar[int] = 9999
    PRODUCTS: ClassVar[dict[int, SyntheticProduct]] = {
        1111: SyntheticProduct("COLD CEREAL", -1.8, 3.5),
        2222: SyntheticProduct("COLD CEREAL", -2.4, 4.2),
        3333: SyntheticProduct("BAG SNACKS", -1.2, 2.5),
        4444: SyntheticProduct("BAG SNACKS", -2.0, 3.0),
        5555: SyntheticProduct("FROZEN PIZZA", -1.5, 5.0),
        9999: SyntheticProduct("ORAL HYGIENE PRODUCTS", -1.0, 2.0, varies=False),
    }
    TITLE: ClassVar[str] = "dunnhumby - Breakfast at the Frat"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def transactions(self) -> pd.DataFrame:
        """Raw transaction rows, including three deliberately dirty rows."""
        weeks = pd.date_range("2009-01-17", periods=self.N_WEEKS, freq="7D")
        grid = pd.MultiIndex.from_product(
            [list(self.PRODUCTS), self.STORES, range(self.N_WEEKS)], names=["UPC", "STORE_NUM", "i"]
        ).to_frame(index=False)
        products = pd.DataFrame(self.PRODUCTS.values(), index=list(self.PRODUCTS))
        p = products.loc[grid["UPC"]].reset_index(drop=True)
        n = len(grid)
        tpr = (self._rng.random(n) < 0.2) & p["varies"]
        feature = tpr & (self._rng.random(n) < 0.5)
        display = self._rng.random(n) < 0.1
        noise = 1 + self._rng.normal(0, 0.03, n) * p["varies"]
        price = p["base_price"] * (1 - 0.2 * tpr) * noise
        mu = 4 + p["elasticity"] * np.log(price / p["base_price"]) + 0.3 * feature + 0.2 * display
        units = self._rng.poisson(np.exp(mu + 0.2 * np.sin(2 * np.pi * grid["i"] / 52)))
        tx = grid.assign(
            WEEK_END_DATE=weeks[grid["i"]], UNITS=units, VISITS=units, HHS=units,
            SPEND=(units * price).round(2), PRICE=price.round(2), BASE_PRICE=p["base_price"],
            FEATURE=feature.astype(int), DISPLAY=display.astype(int), TPR_ONLY=tpr.astype(int),
        ).drop(columns="i")  # fmt: skip
        tx.loc[5, "UNITS"], tx.loc[6, "PRICE"], tx.loc[7, "BASE_PRICE"] = 0, np.nan, np.nan
        return tx

    def products(self) -> pd.DataFrame:
        """Raw product lookup."""
        upcs = list(self.PRODUCTS)
        return pd.DataFrame(
            {
                "UPC": upcs,
                "DESCRIPTION": [f"ITEM {u}" for u in upcs],
                "MANUFACTURER": "ACME",
                "CATEGORY": [p.category for p in self.PRODUCTS.values()],
                "SUB_CATEGORY": "SUB",
                "PRODUCT_SIZE": "10 OZ",
            }
        )

    def stores(self) -> pd.DataFrame:
        """Raw store lookup."""
        return pd.DataFrame(
            {
                "STORE_ID": self.STORES,
                "STORE_NAME": [f"S{s}" for s in self.STORES],
                "ADDRESS_STATE_PROV_CODE": "OH",
                "SEG_VALUE_NAME": "MAINSTREAM",
            }
        )

    def write_workbook(self, path: Path, sheet_names: dict[Sheet, str]) -> Path:
        """Write all sheets with a title row above the header, as the release does."""
        tables = {Sheet.TRANSACTIONS: self.transactions(), Sheet.PRODUCTS: self.products(),
                  Sheet.STORES: self.stores()}  # fmt: skip
        with pd.ExcelWriter(path) as writer:
            for sheet, frame in tables.items():
                title = pd.DataFrame([[self.TITLE]])
                title.to_excel(writer, sheet_name=sheet_names[sheet], header=False, index=False)
                frame.to_excel(writer, sheet_name=sheet_names[sheet], startrow=1, index=False)
        return path


@pytest.fixture
def cfg() -> Config:
    """The shipped config."""
    return Config.load()


@pytest.fixture
def loader_settings(tmp_path: Path, cfg: Config) -> LoaderSettings:
    """Loader settings pointing at a synthetic workbook in tmp_path."""
    raw_path = SyntheticFrat().write_workbook(tmp_path / cfg.data.raw_file, cfg.data.sheets)
    return LoaderSettings(
        raw_path=raw_path,
        cache_dir=tmp_path / "cache",
        sheets=cfg.data.sheets,
        columns=cfg.data.columns,
        header_row=cfg.data.header_row,
    )


@pytest.fixture
def raw(loader_settings: LoaderSettings) -> RawData:
    """Raw tables loaded from the synthetic workbook."""
    return RawDataLoader(loader_settings).load()


@pytest.fixture
def panel(cfg: Config, loader_settings: LoaderSettings, tmp_path: Path) -> pd.DataFrame:
    """Full Phase 1 panel built end to end through the factory."""
    service = DataPrepFactory(cfg, loader_settings, tmp_path / "panel.parquet").build()
    return service.run()


@pytest.fixture(scope="session")
def processed_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Phase 4 and Phase 5 outputs from the synthetic dataset, written like data/processed.

    Runs the real optimization and promo services (stub lift model, short horizon) so
    simulator and app tests read the same files the pipeline writes.
    """
    from src.config import ModelName
    from src.optimization import OptimizationFactory, OptimizationOutputs
    from src.optimization.inputs import PRODUCT_ATTRS
    from src.promo import PromoFactory, PromoOutputs
    from src.schema import Col, ElasticitySource
    from src.simulator.data import SOURCES
    from tests.promo.conftest import LinearLift

    root = tmp_path_factory.mktemp("processed")
    base = Config.load()
    settings = LoaderSettings(
        raw_path=SyntheticFrat().write_workbook(root / base.data.raw_file, base.data.sheets),
        cache_dir=root / "cache",
        sheets=base.data.sheets,
        columns=base.data.columns,
        header_row=base.data.header_row,
    )
    panel = DataPrepFactory(base, settings, root / "panel.parquet").build().run()
    cfg = base.model_copy(
        update={
            "optimization": base.optimization.model_copy(update={"robustness_draws": 3}),
            "promo": base.promo.model_copy(update={"frontier_points": 4, "budget": 20_000.0}),
            "forecast": base.forecast.model_copy(update={"horizon_weeks": 8}),
        }
    )
    baseline = panel.assign(**{Col.BASELINE: panel[Col.UNITS] * 0.8})
    final = (
        panel.drop_duplicates(Col.UPC)[list(PRODUCT_ATTRS)]
        .assign(
            **{
                Col.ELASTICITY: -1.8,
                Col.ELASTICITY_LOW: -2.3,
                Col.ELASTICITY_HIGH: -1.3,
                Col.ELASTICITY_SOURCE: ElasticitySource.ESTIMATED.value,
                Col.MODEL: ModelName.BAYES_HIERARCHICAL.value,
                Col.PROMO_ELASTICITY: -2.5,
            }
        )
        .reset_index(drop=True)
    )
    name = {field: path.name for field, (path, _) in SOURCES.items()}
    outputs = OptimizationOutputs(
        root / name["recommendations"], root / name["optimization_summary"]
    )
    report = OptimizationFactory(cfg, outputs).build().run(baseline, final, pd.DataFrame())
    promo_outputs = PromoOutputs(
        *(root / name[f] for f in ("forecast", "backtest", "candidates", "plans", "frontier",
                                   "promo_summary"))
    )  # fmt: skip
    models = {ModelName.BAYES_HIERARCHICAL: LinearLift()}
    PromoFactory(cfg, models, promo_outputs).build().run(
        panel, baseline, final, report.recommendations
    )
    return root
