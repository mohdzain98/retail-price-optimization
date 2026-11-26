"""Tests for src.utils: the generic Registry and the exception hierarchy.

Registry backs every extension point, so ordering, lookup and duplicate
handling are checked directly. Exceptions must share one base class.
One test class per production class.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from src.utils import ConfigError, DataValidationError, MissingFileError, Registry, RPOError


class TestRegistry:
    """Registry stores classes by key in registration order."""

    def test_register_get_and_order(self) -> None:
        """Lookup by key works and iteration follows registration order."""
        reg: Registry[str, type] = Registry("key")

        @reg.register
        class B:
            key: ClassVar[str] = "b"

        @reg.register
        class A:
            key: ClassVar[str] = "a"

        assert reg.get("a") is A and list(reg) == [B, A] and reg.keys() == ["b", "a"]
        assert len(reg) == 2

    def test_duplicate_and_unknown_keys_raise(self) -> None:
        """Duplicate registration and unknown lookups raise ConfigError."""
        reg: Registry[str, type] = Registry("key")

        @reg.register
        class One:
            key: ClassVar[str] = "x"

        class Two:
            key: ClassVar[str] = "x"

        with pytest.raises(ConfigError, match="Duplicate"):
            reg.register(Two)
        with pytest.raises(ConfigError, match="registered"):
            reg.get("missing")


class TestExceptions:
    """All project errors share RPOError as base."""

    @pytest.mark.parametrize("error", [ConfigError, MissingFileError, DataValidationError])
    def test_hierarchy(self, error: type[Exception]) -> None:
        """Each custom error is an RPOError."""
        assert issubclass(error, RPOError)
