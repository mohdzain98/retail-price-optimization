"""Custom exception hierarchy for the whole project.

Every project error inherits from RPOError so callers can catch one base type.
Raise the most specific subclass; never raise bare Exception or swallow errors.
Lives in utils because every layer raises these.
"""


class RPOError(Exception):
    """Base class for all retail-price-optimization errors.

    Catch this at entry points (CLI, app) to report project failures cleanly.
    """


class ConfigError(RPOError):
    """Configuration is invalid or inconsistent with the data.

    Example: a holdout longer than the series, or an unknown registry key.
    """


class MissingFileError(RPOError):
    """A required input file (raw dataset, cached output) does not exist.

    Message should tell the user how to produce or obtain the file.
    """


class DataValidationError(RPOError):
    """Input data does not match the expected schema or invariants.

    Example: a mapped column or sheet is missing from the raw workbook.
    """
