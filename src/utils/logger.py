"""Logging toolkit for the app: setup, formatters and a switchable pretty-print.

AppLogger.init() configures logging once (call it at the top of app.py) and
AppLogger.get_logger(__name__) returns a logger that also has .print().
logger.print outputs only when RPO_PRINT_ENABLED is true; RPO_DEBUG sets DEBUG level.
"""

import contextvars
import json
import logging
import os
import uuid
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from rich.console import Console
from rich.pretty import pretty_repr

ENV_PRINT_ENABLED = "RPO_PRINT_ENABLED"
ENV_DEBUG = "RPO_DEBUG"
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
}
RESET_COLOR = "\033[0m"
LEVEL_WIDTH = 9

CONSOLE_FORMAT = "%(colored_levelname)s %(name)s | %(message)s%(extra_suffix)s"
FILE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | "
    "%(message)s%(extra_suffix)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGS_DIR = Path(__file__).resolve().parents[2] / ".cache" / "logs"
DEFAULT_LOG_FILE = "rpo.log"
API_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")

NO_REQUEST_ID = "-"
REQUEST_ID_LENGTH = 8
STANDARD_RECORD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__)
INTERNAL_FIELDS = frozenset({"colored_levelname", "error_suffix", "extra_suffix"})

# Request ID of the call being served; shown on every log line while it is set.
request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=NO_REQUEST_ID
)
_console = Console()


def new_request_id() -> str:
    """Set and return a fresh short request ID."""
    rid = uuid.uuid4().hex[:REQUEST_ID_LENGTH]
    request_id.set(rid)
    return rid


def _flag(name: str) -> bool:
    """Return True when the environment variable is set to a true-like value."""
    return os.environ.get(name, "").strip().lower() in TRUE_VALUES


def _format_extra_value(value: object) -> str:
    """Render one extra value as text: strings as-is, others as JSON when possible."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


class SafeExtraFormatter(logging.Formatter):
    """Formatter that appends the record's extra fields as key=value pairs.

    Format strings must include %(extra_suffix)s to show them.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the record, first building its extra_suffix."""
        items = []
        if request_id.get() != NO_REQUEST_ID and not hasattr(record, "request_id"):
            items.append(f"request_id={request_id.get()}")
        for key, value in sorted(record.__dict__.items()):
            if key in STANDARD_RECORD_FIELDS or key in INTERNAL_FIELDS or key.startswith("_"):
                continue
            items.append(f"{key}={_format_extra_value(value)}")
        record.extra_suffix = f" | {' | '.join(items)}" if items else ""
        return super().format(record)


class ColorFormatter(SafeExtraFormatter):
    """Console formatter that sets colored_levelname, a padded ANSI-coloured level.

    Format strings use %(colored_levelname)s to show it.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the record with a coloured level name."""
        padded = f"{record.levelname + ':':<{LEVEL_WIDTH}}"
        color = LEVEL_COLORS.get(record.levelname, "")
        record.colored_levelname = f"{color}{padded}{RESET_COLOR}" if color else padded
        return super().format(record)


class DebugAwareLogger(logging.Logger):
    """Logger whose print() is a no-op unless RPO_PRINT_ENABLED is true.

    The switch is read live and is independent of RPO_DEBUG (the log level).
    Never pass raw data rows: print shapes, counts and parameters only.
    """

    def print(self, label: str, data: object = None, *, raw_keys: set[str] | None = None) -> None:
        """Pretty-print data under a label rule when printing is enabled.

        raw_keys names dict keys whose string value prints as literal text (real newlines).
        """
        if not _flag(ENV_PRINT_ENABLED):
            return
        _console.rule(f"[bold cyan]{label}[/bold cyan]", style="cyan")
        if data is None:
            return
        if isinstance(data, dict) and raw_keys:
            data = self._print_raw_keys(data, raw_keys)
            if not data:
                return
        _console.print(pretty_repr(data))

    @staticmethod
    def _print_raw_keys(data: dict[str, object], raw_keys: set[str]) -> dict[str, object]:
        """Print the raw_keys string values as text and return the remaining items."""
        remaining: dict[str, object] = {}
        for key, value in data.items():
            if key in raw_keys and isinstance(value, str):
                _console.print(f"[bold]{key}:[/bold]")
                _console.print(value)
            else:
                remaining[key] = value
        return remaining


# Must run before any getLogger() call so every logger is a DebugAwareLogger.
logging.setLoggerClass(DebugAwareLogger)


class AppLogger:
    """Configures logging once per process and hands out named loggers.

    Call AppLogger.init() at the top of app.py, before the first get_logger().
    """

    _configured: bool = False

    @classmethod
    def init(
        cls,
        level: int | None = None,
        log_to_file: bool = False,
        filename: str = DEFAULT_LOG_FILE,
    ) -> None:
        """Set up console (and optional file) logging; later calls do nothing.

        Level defaults to DEBUG when RPO_DEBUG is true, else INFO. Reads .env first.
        """
        if cls._configured:
            return
        cls._configured = True
        load_dotenv()
        level = level if level is not None else cls._default_level()
        logging.setLoggerClass(DebugAwareLogger)
        root = logging.getLogger()
        root.setLevel(level)
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        root.addHandler(cls._console_handler(level))
        for name in API_LOGGERS:
            logging.getLogger(name).setLevel(level)
        if log_to_file:
            root.addHandler(cls._file_handler(filename))

    @staticmethod
    def _default_level() -> int:
        """DEBUG when RPO_DEBUG is true, else INFO."""
        return logging.DEBUG if _flag(ENV_DEBUG) else logging.INFO

    @staticmethod
    def _console_handler(level: int) -> logging.Handler:
        """Build the coloured console handler."""
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(ColorFormatter(CONSOLE_FORMAT))
        return handler

    @staticmethod
    def _file_handler(filename: str) -> logging.Handler:
        """Build the warnings-and-above file handler under .cache/logs."""
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(LOGS_DIR / filename, encoding="utf-8")
        handler.setLevel(logging.WARNING)
        handler.setFormatter(SafeExtraFormatter(FILE_FORMAT, datefmt=DATE_FORMAT))
        return handler

    @staticmethod
    def get_logger(name: str | None = None) -> DebugAwareLogger:
        """Return a logger named after the last part of name (usually __name__)."""
        short_name = (name if name is not None else __name__).split(".")[-1]
        return cast(DebugAwareLogger, logging.getLogger(short_name))
