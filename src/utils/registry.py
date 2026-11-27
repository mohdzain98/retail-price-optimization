"""Generic ordered registry backing every Open/Closed extension point.

Implementations register with a class decorator under a key read from a class
attribute; callers look classes up by key or iterate in registration order.
Used for aggregators, product filters, feature steps and pipeline stages.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator

from src.utils.exceptions import ConfigError


class Registry[K: Hashable, T: type]:
    """Maps a key (taken from `key_attr` on each class) to a registered class.

    Keeps registration order, so iteration order is definition order.
    """

    def __init__(self, key_attr: str) -> None:
        self._key_attr = key_attr
        self._items: dict[K, T] = {}

    def register(self, cls: T) -> T:
        """Class decorator: store cls under its key attribute.

        Raises ConfigError on a duplicate key.
        """
        key: K = getattr(cls, self._key_attr)
        if key in self._items:
            raise ConfigError(f"Duplicate registry key {key!r} for {cls.__name__}")
        self._items[key] = cls
        return cls

    def get(self, key: K) -> T:
        """Return the class registered under key.

        Raises ConfigError listing the valid keys when key is unknown.
        """
        if key not in self._items:
            raise ConfigError(f"Unknown key {key!r}; registered: {list(self._items)}")
        return self._items[key]

    def keys(self) -> list[K]:
        """Registered keys in registration order."""
        return list(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)
