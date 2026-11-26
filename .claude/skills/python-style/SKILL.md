---
name: python-style
description: Python code style conventions for this project — type hints, docstring length/format, comment brevity, class-based structure with one-line imports, DRY, Open/Closed and SOLID rules, data objects, errors/logging, size limits, pandas and testing conventions. Apply this whenever writing or editing any Python file (.py), including new modules, new classes, new functions/methods, and edits to existing code. Always follow this skill for Python code in this project, not just when explicitly asked about style.
---

# Python Style

Style conventions for all Python code written or edited in this project. Apply them automatically; do not wait to be asked.

## Type hints: mandatory

Every function and method signature includes type hints for **all parameters and the return type**, with no exceptions (including `-> None`).

```python
def find_record(record_id: int, include_archived: bool = False) -> Record | None: ...


async def fetch_page(self, url: str, timeout_s: float = 10.0) -> Page: ...


def log_event(self, name: str, data: dict | None = None) -> None: ...
```

## Comments: short, not narrative

Inline comments are brief, a phrase rather than a sentence, and appear only where the *why* isn't obvious from the code itself. Never comment what the code already says.

```python
# Good
retries = 3  # API occasionally rate-limits on cold start

# Bad: restates the obvious
retries = 3  # set retries to 3

# Bad: too long, reads like a paragraph
retries = 3  # We set this to 3 because in our testing we found that the API
# tends to rate-limit us when the connection is cold, and 3 retries
# was generally enough to get through without excessive delay.
```

Skip comments entirely when the code is self-explanatory. Prefer a clear variable or function name over a comment explaining an unclear one.

## Module docstrings: 4-5 lines

Every `.py` file starts with a module-level docstring of 4-5 lines: what the module contains and its role in the system. No more, no less. Trim to fit; don't pad to fit.

```python
"""Loaders that read raw source files into validated tables.

Handles file discovery, header detection and column renaming, and caches
parsed tables so later runs skip the slow read. Used by the data-prep
service; nothing else should read raw files directly.
"""
```

## Class docstrings: 3-4 lines

Every class has a docstring of 3-4 lines: what it represents or does, and any non-obvious responsibility it owns.

```python
class CachedFileLoader(FileLoader):
    """Reads a source file into a DataFrame, using a parquet cache.

    Owns cache invalidation: the cache is rebuilt whenever the source
    file is newer than the cached copy.
    """
```

## Method/function docstrings: 2-3 lines

Every method and function has a docstring of 2-3 lines: what it does and anything a caller needs to know (side effects, non-obvious params, when it returns None or raises).

```python
def resolve_deadline(duration_hint: str, created_at: datetime) -> datetime | None:
    """Convert a duration hint into an absolute deadline.

    Returns None for "open_ended" hints, which never expire on their own.
    """
```

**Exception**: trivial one-line accessors, `__init__` methods that just assign params, and dunder methods don't need a docstring unless they do something non-obvious.

## Length discipline

If a docstring runs longer than the line counts above, the function is probably doing too much. Consider splitting it rather than writing a longer docstring to explain it.

## Structure: classes and one-line imports

Behaviour lives in classes, not loose module-level functions.

- Keep the related classes and helpers of one domain together in one file; do not split one class per file. A folder is only for **two or more files of the same type** (for example `src/prep/` holding several data-prep modules). A lone file stays a plain module (`src/config.py`), never a folder with one file. Helpers used only inside a file start with `_`.
- Each package `__init__.py` re-exports its public classes and defines `__all__`, so callers import in one line.
- Each layer exposes one facade class, and other layers import only that facade. Wiring (building the objects a facade needs) happens in one composition root, such as a factory or the CLI entry point.
- Pass dependencies in through `__init__` (dependency injection); never build them inside the class. Prefer composition over inheritance, and keep inheritance to one level.
- Pure stateless helpers (decorators, small generic utilities) may stay as functions in `src/utils/`.

```python
# src/<layer>/__init__.py
from src.<layer>.service import LayerService

__all__ = ["LayerService"]

# elsewhere: one line, with the src prefix
from src.<layer> import LayerService
```

## DRY

- Rule of three: tolerate a second copy, and extract on the third (base class, mixin or helper).
- No magic numbers or strings. Use a settings object, named constants or an `Enum` (for example `Status.ACTIVE = "active"`).
- Define each column name, format and default once, in one place, and reference it.

## Open/Closed

Extend by adding a class, never by editing an existing class to add a case.

- Define an abstract base or `Protocol` for each extension point (for example `Step`, `Source`, `Exporter`).
- New implementations register themselves in a registry (decorator or dict), and the caller loops over the registry.
- No `if/elif` chains that branch on a type or name; use polymorphism or a lookup dict.

```python
class Step(ABC):
    """Base for every pipeline step; subclasses implement apply only."""

    @abstractmethod
    def apply(self, frame: pd.DataFrame) -> pd.DataFrame: ...


@STEPS.register
class DropDuplicates(Step):
    """Removes duplicate rows on the configured key columns."""

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame: ...
```

## Other SOLID rules

- Single responsibility: each class has one reason to change. If a class needs "and" to describe it, split it.
- Depend on abstractions (ABC/Protocol), not concrete classes, across layer boundaries.
- Layer rule: lower layers never import higher ones. For example, data prep never imports the pipeline CLI or the app.

## Data objects

- Params and results are `@dataclass(frozen=True)` or frozen pydantic models, not raw dicts or tuples.
- Fixed choices are `Enum`s, not strings.
- Do not mutate inputs; work on a copy or return new objects.

## Errors and logging

- Custom exceptions live in one module (for example `ConfigError`, `MissingFileError`, `DataValidationError`), all inheriting from one base error.
- No bare `except:`. Catch the narrowest type, and never swallow errors silently.
- Use `logger`, never `print`, and never log secrets or data rows.

## Size and naming limits

- Functions about 30 lines or fewer, files about 300 lines or fewer, and at most 5 parameters (group extras into a params object).
- `PascalCase` classes, `snake_case` functions and variables, `UPPER_SNAKE` constants. Names say what the thing is (`days_since_last_sale`, not `d`).
- Use `pathlib.Path`, never string paths. Read paths from config, not inside functions.
- Absolute imports only, ordered standard library, then third party, then local. No `import *`.
- Project imports always carry the `src.` prefix (`from src.config import Config`, `from src.utils import AppLogger`), so the origin is visible.

## Pandas

- Vectorize. Avoid `iterrows` or row-wise `apply` where a column operation works.
- Reference columns through constants or enums, not repeated string literals.
- Return new DataFrames; do not modify the argument in place.

## Tests and tooling

- One test class per production class, asserting known reference values where they exist.
- Code must pass `ruff` (lint and format) and `mypy` (types).

## Applying this to existing code

When editing an existing file that doesn't yet follow these conventions, bring the function, class or module you're touching into compliance as part of the edit. Don't do a drive-by reformat of the whole file for an unrelated change; only fix the code you're already modifying.
