## What and why

<!-- What does this PR change, and why? Link the issue, e.g. "Closes #12". -->

## How it was checked

<!-- Tests added or changed, commands run, before/after numbers if results move. -->

## Checklist

- [ ] The linked issue has my comment that I'm working on it.
- [ ] One topic per PR, branched from and targeting `main`.
- [ ] `make check` passes (ruff, mypy, pytest, pre-commit).
- [ ] `make pr-check` passes: no data, secrets, local notes or files over 2 MB.
- [ ] Tests added or updated for every behaviour change (they run on synthetic data, not the licensed dataset).
- [ ] Logic lives in `src/`; notebooks and Streamlit pages only call it.
- [ ] Type hints on every signature; docstrings follow the project style.
- [ ] New behaviour goes through a registry (model, constraint, forecaster, planner, …), not an `if/elif` chain.
- [ ] Docs updated if needed: `docs/methodology.md` (what is computed), `docs/architecture.md` (where code lives),
      `docs/data.md` (output tables).
- [ ] If results change: figures regenerated (`reports/figures/`) and README numbers updated.
