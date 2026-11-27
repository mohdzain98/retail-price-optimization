"""Persistence of fitted elasticity models between pipeline stages.

The elasticity stage saves every fitted model; the baseline stage loads them to
decompose volume with each product's selected model. Pickle is used because the
models hold pandas/numpy state only (no live PyMC graph); regenerate on code change.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from src.config import ModelName
from src.elasticity.base import ElasticityModel
from src.utils import AppLogger, MissingFileError

logger = AppLogger.get_logger(__name__)


class ModelStore:
    """Saves and loads the fitted models as one {ModelName: model} pickle.

    Only load files this pipeline wrote: pickle executes code on load.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def save(self, models: dict[ModelName, ElasticityModel]) -> None:
        """Write all fitted models."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("wb") as handle:
            pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved fitted models", extra={"models": [m.value for m in models]})

    def load(self) -> dict[ModelName, ElasticityModel]:
        """Read fitted models; raises MissingFileError if the elasticity stage has not run."""
        if not self._path.exists():
            raise MissingFileError(f"{self._path} not found; run `make elasticity` first")
        with self._path.open("rb") as handle:
            models: dict[ModelName, ElasticityModel] = pickle.load(handle)  # noqa: S301 (own artifact)
        return models
