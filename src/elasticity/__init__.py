"""Elasticity layer (Phase 2): per-product own-price elasticity with guard-rails.

base.py (spec, model base + registry, scorer), frequentist.py (OLS, pooled,
MixedLM), bayes.py (PyMC hierarchical), guardrails.py, service.py (facade).
Importing this package registers every model: from src.elasticity import ElasticityFactory.
"""

from src.elasticity import bayes, frequentist  # noqa: F401  (imports register the models)
from src.elasticity.service import (
    ElasticityFactory,
    ElasticityOutputs,
    ElasticityReport,
    ElasticityService,
)

__all__ = ["ElasticityFactory", "ElasticityOutputs", "ElasticityReport", "ElasticityService"]
