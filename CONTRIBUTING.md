# Contributing

Thanks for your interest in improving this project. Issues and pull requests are welcome: bug reports, clearer docs,
new elasticity models, constraints, forecasters, promo planners or simulator features.

## Getting set up

1. Fork and clone the repository, then run `make setup`. It uses [uv](https://docs.astral.sh/uv/) when installed,
   otherwise `python3 -m venv` + pip. You need Python 3.12.
2. The dunnhumby *Breakfast at the Frat* dataset is licensed and is **never committed**. To run the pipeline on real
   data, download it yourself (see [docs/data.md](docs/data.md)). The test suite does **not** need it: it runs on a
   synthetic dataset.

## Workflow

- Branch from `main` (for example `feature/cross-price-effects` or `fix/frontier-hurdle`) and open the pull request
  against `main`.
- Keep a pull request to one topic, and describe what changed and how you checked it.
- Run `make check` and `make pr-check` before pushing. CI runs the same on every pull request into `main`:
  - **File guard:** the PR must not add data (the `data/` folder, `.parquet`, `.csv`, `.xlsx`, `.pkl`, `.db`, …),
    secrets (`.env`, `.streamlit/secrets.toml`), local notes or caches, or any file over 2 MB.
  - **Pre-commit hooks**, ruff (lint and format), mypy and pytest.

## Code conventions

- **Logic lives in `src/`.** Notebooks and Streamlit pages only call it, plot and narrate. Notebook outputs are
  stripped on commit (`nbstripout`).
- **Type hints on every signature**, including `-> None`.
- **Docstrings:** 4–5 lines per module, 3–4 per class, 2–3 per method. Comments are short and say *why*, not *what*.
- **Extend through registries instead of `if/elif` chains.** To add a model, constraint, forecaster or planner,
  subclass the base class and register it. The registries are listed in
  [docs/architecture.md](docs/architecture.md#extension-points-openclosed).
- **No magic strings or numbers.** Column names come from `Col` (`src/schema.py`), and tunables from
  `config/config.yaml` through the frozen `Config`.
- **Errors** derive from `RPOError`. Log through `AppLogger`; there are no `print` calls in `src/`.
- A folder exists only when it holds two or more modules of the same kind.

## Tests

- Tests mirror `src/`, with one test class per production class. Add or update tests with every behaviour change.
- Prefer known-answer tests: closed-form optima, hand-computed economics, synthetic data with a known elasticity.
- Streamlit pages are tested in-process with `AppTest` against synthetic pipeline outputs (`tests/app/`).

## Docs

If you change what a step computes, update [docs/methodology.md](docs/methodology.md). If you change where code lives
or how stages connect, update [docs/architecture.md](docs/architecture.md). If you add or change an output table,
update [docs/data.md](docs/data.md).

## Never commit

- Raw or processed data (`data/raw/`, `data/processed/`), or the local accounts file `data/users.db`.
- Secrets: `.env`, `.streamlit/secrets.toml`.

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
