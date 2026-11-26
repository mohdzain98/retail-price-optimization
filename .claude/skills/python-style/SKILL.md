---
name: python-style
description: Python code style conventions for this project — docstring length/format for modules, classes, and methods, comment brevity, and mandatory type hints on function signatures. Apply this whenever writing or editing any Python file (.py), including new modules, new classes, new functions/methods, and edits to existing code. Always follow this skill for Python code in this project, not just when explicitly asked about style.
---

# Python Style

Style conventions for all Python code written or edited in this project. Apply automatically — do not wait to be asked.

## Type hints — mandatory

Every function and method signature includes type hints for **all parameters and the return type**, with no exceptions (including `-> None`).

```python
def get_user_profile(user_id: UUID, include_facts: bool = False) -> UserProfile | None: ...


async def save_turn(self, chat_id: UUID, user_message: str) -> ChatTurn: ...


def log_event(self, name: str, data: dict | None = None) -> None: ...
```

## Comments — short, not narrative

Inline comments are brief — a phrase, not a sentence, and only where the *why* isn't obvious from the code itself. Never comment what the code already says.

```python
# Good
retries = 3  # API occasionally rate-limits on cold start

# Bad — restates the obvious
retries = 3  # set retries to 3

# Bad — too long, reads like a paragraph
retries = 3  # We set this to 3 because in our testing we found that the API
# tends to rate-limit us when the connection is cold, and 3 retries
# was generally enough to get through without excessive delay.
```

Skip comments entirely when the code is self-explanatory. Prefer a clear variable/function name over a comment explaining an unclear one.

## Module docstrings — 4-5 lines

Every `.py` file starts with a module-level docstring, 4-5 lines: what the module contains, and its role in the system. No more, no less — trim to fit, don't pad to fit.

```python
"""Repository layer for chat turns and their embedded child chunks.

Handles persistence of ChatTurn (parent) and ChatMemoryChunk (child) records,
including the parent-child chunking flow used for semantic memory search.
Used by the memory tools and the profile extraction pipeline.
"""
```

## Class docstrings — 3-4 lines

Every class has a docstring, 3-4 lines: what it represents/does and any non-obvious responsibility it owns.

```python
class MemoryRepository(BaseRepository[ChatMemoryChunk]):
    """Persists and searches chat memory chunks scoped by user_id.

    Owns the chunk/embed/store flow on write, and the over-fetch-then-
    deduplicate pattern on similarity search to avoid duplicate parent turns.
    """
```

## Method/function docstrings — 2-3 lines

Every method and function has a docstring, 2-3 lines: what it does and anything a caller needs to know (side effects, non-obvious params, when it returns None/raises).

```python
def resolve_expiry(duration_hint: str, created_at: datetime) -> datetime | None:
    """Convert a duration hint into an expires_at timestamp.

    Returns None for "long_term" facts, which never expire on their own.
    """
```

**Exception**: trivial one-line accessors, `__init__` methods that just assign params, and dunder methods don't need a docstring unless they do something non-obvious.

## Length discipline

If a docstring is running longer than the line counts above, that's a signal the function is doing too much — consider splitting it, rather than writing a longer docstring to explain it.

## Structure — classes and one-line imports

Behaviour lives in classes, not loose module-level functions.

- Keep related classes and helpers of one domain together in one file; do not split one class per file. A folder is for several files of one type (for example `core/metrics/`, `core/tables/`); a single file stays a plain module (`utility/logger.py`). Helpers used only inside a file start with `_`.
- Each package `__init__.py` re-exports its public classes and defines `__all__`, so callers import in one line.
- Each layer exposes one facade class. Other layers import only that facade.
- Pass dependencies in through `__init__` (dependency injection); never build them inside the class. Prefer composition over inheritance, and keep inheritance to one level.
- Pure utilities with no state may stay as functions in the `utility` package.

```python
# src/core/__init__.py
from src.core.services import MetricService

__all__ = ["MetricService"]

# elsewhere: one line, with the src prefix
from src.core import MetricService
```

## DRY

- Rule of three: tolerate a second copy, extract on the third (base class, mixin or helper).
- No magic numbers or strings. Use a `Settings` class, constants or an `Enum` (for example `MovementType.ORDER_ISSUE = "261"`).
- Define each column name, format and default once, in one place, and reference it.

## Open/Closed

Extend by adding a class, never by editing an existing class to add a case.

- Define an abstract base or `Protocol` for each extension point (`Metric`, `DataSource`, `Exporter`).
- New implementations register themselves in a registry (decorator or dict). The caller loops over the registry.
- No `if/elif` chains that branch on a type or name; use polymorphism or a lookup dict.

```python
class Metric(ABC):
    """Base for every metric; subclasses implement compute only."""

    @abstractmethod
    def compute(self, params: Params) -> pd.DataFrame: ...


@register_metric
class SlowMovingMetric(Metric):
    """Parts with no consumption for N years."""

    def compute(self, params: Params) -> pd.DataFrame: ...
```

## Other SOLID rules

- Single responsibility: one reason to change per class. If a class needs "and" to describe it, split it.
- Depend on abstractions (ABC/Protocol), not concrete classes, across layer boundaries.
- Layer rule: lower layers never import higher ones (for example `core` never imports `ui` or `api`).

## Data objects

- Params and results are `@dataclass(frozen=True)` or pydantic models, not raw dicts or tuples.
- Fixed choices are `Enum`s, not strings.
- Do not mutate inputs; work on a copy or return new objects.

## Errors and logging

- Custom exceptions live in one module (`DataValidationError`, `MissingFileError`), all inheriting from one base error.
- No bare `except:`; catch the narrowest type; never swallow errors silently.
- Use `logger`. Never `print`. Never log secrets or data rows.

## Size and naming limits

- Functions about 30 lines or fewer, files about 300 lines or fewer, at most 5 parameters (group extras into a params object).
- `PascalCase` classes, `snake_case` functions and variables, `UPPER_SNAKE` constants. Names say what the thing is (`days_since_consumed`, not `d`).
- Use `pathlib.Path`, never string paths. Read paths from config, not inside functions.
- Absolute imports only, ordered standard library, third party, local. No `import *`.
- Project imports always carry the `src.` prefix (`from src.core import MetricService`, `from src.utility.logger import AppLogger`) so the origin is visible. Never use bare `core`, `ui` or `utility`.

## Pandas

- Vectorize; no `iterrows` or row-wise `apply` where a column operation works.
- Reference columns through constants or enums, not repeated string literals.
- Return new DataFrames; do not modify the argument in place.

## Tests and tooling

- One test class per production class; assert known reference values where they exist.
- Code must pass `ruff` (lint and format) and `mypy` (types).

## Applying this to existing code

When editing an existing file that doesn't yet follow these conventions, bring the function/class/module you're touching into compliance as part of the edit. Don't do a drive-by reformat of the whole file for an unrelated change — only the code you're already modifying.
