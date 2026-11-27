"""Fixtures for app tests: undo AppTest's replacement of the __main__ module.

AppTest runs the app script as `__main__`; multiprocessing's spawn start method
(PyMC sampling) re-imports `__main__` in every child, which would start the app
there and hang. Restoring the module after each test keeps later tests safe.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def restore_main_module() -> Iterator[None]:
    """Put the original __main__ module back after each app test."""
    main = sys.modules["__main__"]
    yield
    sys.modules["__main__"] = main
