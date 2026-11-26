"""Tests for src.elasticity.service: the facade runs end to end and writes outputs.

Uses the two fast models (per-product and pooled OLS) so the run is quick;
the Bayesian model is covered in test_bayes.py.
One test class per production class.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config, ModelName
from src.elasticity import ElasticityFactory, ElasticityOutputs, ElasticityService
from src.schema import Col, ImputeReason
from tests.elasticity.conftest import SyntheticPanel


def _outputs(tmp_path: Path) -> ElasticityOutputs:
    """Output paths under tmp_path."""
    return ElasticityOutputs(
        tmp_path / "final.parquet", tmp_path / "all.parquet", tmp_path / "cmp.parquet", tmp_path
    )


def _fast_config() -> Config:
    """Shipped config restricted to the fast models."""
    cfg = Config.load()
    models = [ModelName.OLS_PER_PRODUCT, ModelName.POOLED_OLS]
    elasticity = cfg.elasticity.model_copy(update={"models": models, "model_priority": models})
    return cfg.model_copy(update={"elasticity": elasticity})


class TestElasticityService:
    """ElasticityService fits, scores, selects, imputes and writes."""

    def test_run_writes_outputs(self, synthetic_panel: pd.DataFrame, tmp_path: Path) -> None:
        """All tables are written; eligible products use OLS, the ineligible one is imputed."""
        outputs = _outputs(tmp_path)
        report = ElasticityFactory(_fast_config(), outputs).build().run(synthetic_panel)
        assert all(p.exists() for p in (outputs.final, outputs.all_models, outputs.comparison))
        assert len(report.comparison) == 2 and (report.comparison[Col.WAPE] < 0.2).all()
        final = report.final.set_index(Col.UPC)
        assert len(final) == synthetic_panel[Col.UPC].nunique()
        assert (
            final.loc[SyntheticPanel.INELIGIBLE_UPC, Col.IMPUTE_REASON] == ImputeReason.INELIGIBLE
        )
        eligible = final.drop(SyntheticPanel.INELIGIBLE_UPC)
        assert (eligible[Col.MODEL] == ModelName.OLS_PER_PRODUCT.value).all()


class TestElasticityFactory:
    """ElasticityFactory wires a ready-to-run service."""

    def test_builds_service(self, tmp_path: Path) -> None:
        """Returns an ElasticityService for the shipped config."""
        assert isinstance(
            ElasticityFactory(Config.load(), _outputs(tmp_path)).build(), ElasticityService
        )
