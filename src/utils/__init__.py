"""Cross-cutting helpers with no business logic.

logger.py (AppLogger), exceptions.py (project error hierarchy) and registry.py
(generic Open/Closed registry). Import in one line, for example:
from src.utils import AppLogger, Registry, DataValidationError.
"""

from src.utils.exceptions import (
    ConfigError,
    DataValidationError,
    MissingFileError,
    RPOError,
    SolverError,
)
from src.utils.logger import AppLogger
from src.utils.registry import Registry

__all__ = [
    "AppLogger",
    "ConfigError",
    "DataValidationError",
    "MissingFileError",
    "RPOError",
    "Registry",
    "SolverError",
]
