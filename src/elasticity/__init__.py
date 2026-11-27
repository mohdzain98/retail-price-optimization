"""Elasticity layer (Phase 2): per-product own-price elasticity with guard-rails.

base.py (spec, model base + registry, scorer), frequentist.py (OLS, pooled,
MixedLM), bayes.py (PyMC), guardrails.py, store.py (model persistence), service.py.
Importing this package registers every model: from src.elasticity import ElasticityFactory.
"""

from src.elasticity import bayes, frequentist  # noqa: F401  (imports register the models)
from src.elasticity.base import ElasticityModel
from src.elasticity.service import (
    ElasticityFactory,
    ElasticityOutputs,
    ElasticityReport,
    ElasticityService,
)
from src.elasticity.store import ModelStore

__all__ = [
    "ElasticityFactory",
    "ElasticityModel",
    "ElasticityOutputs",
    "ElasticityReport",
    "ElasticityService",
    "ModelStore",
]
