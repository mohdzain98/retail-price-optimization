"""Baseline layer (Phase 3): top-down baseline and due-to decomposition.

effects.py (effect terms with each product's model), rules.py (bad-baseline rules
and corrector), decomposition.py (incremental lifts, product summary), service.py
(facade + factory). Import in one line: from src.baseline import BaselineFactory.
"""

from src.baseline.service import (
    BaselineFactory,
    BaselineOutputs,
    BaselineReport,
    BaselineService,
)

__all__ = ["BaselineFactory", "BaselineOutputs", "BaselineReport", "BaselineService"]
