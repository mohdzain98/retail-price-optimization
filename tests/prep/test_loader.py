"""Tests for src.prep.loader: header detection, renaming, caching and errors.

Uses the synthetic workbook (title row above the header) from conftest.
One test class per production class: RawDataLoader and LoaderSettings.
Covers both the happy path and each loud-failure mode.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.config import DATA_RAW, Config, Sheet
from src.prep import LoaderSettings, RawDataLoader
from src.schema import Col
from src.utils import DataValidationError, MissingFileError


class TestRawDataLoader:
    """RawDataLoader reads, renames, caches and fails loudly."""

    def test_detects_header_and_renames(self, loader_settings: LoaderSettings) -> None:
        """Title row is skipped and columns become canonical Col names."""
        raw = RawDataLoader(loader_settings).load()
        assert {Col.WEEK, Col.STORE, Col.UPC, Col.UNITS, Col.TPR} <= set(raw.transactions.columns)
        assert pd.api.types.is_datetime64_any_dtype(raw.transactions[Col.WEEK])
        assert Col.CATEGORY in raw.products.columns and Col.STORE in raw.stores.columns

    def test_second_load_reads_identical_cache(self, loader_settings: LoaderSettings) -> None:
        """Cache files are written and reproduce the Excel read exactly."""
        loader = RawDataLoader(loader_settings)
        first = loader.load()
        assert len(list(loader_settings.cache_dir.glob("raw_*.parquet"))) == len(Sheet)
        pd.testing.assert_frame_equal(first.transactions, loader.load().transactions)

    def test_missing_column_raises(self, loader_settings: LoaderSettings) -> None:
        """A mapped column absent from the sheet names the column in the error."""
        columns = {**loader_settings.columns, Sheet.PRODUCTS: {"UPC": Col.UPC, "NOPE": Col.STATE}}
        loader = RawDataLoader(replace(loader_settings, columns=columns))
        with pytest.raises(DataValidationError, match="NOPE"):
            loader.load(use_cache=False)

    def test_missing_sheet_raises(self, loader_settings: LoaderSettings) -> None:
        """A wrong sheet name is a DataValidationError, not a raw ValueError."""
        sheets = {**loader_settings.sheets, Sheet.STORES: "No Such Sheet"}
        with pytest.raises(DataValidationError, match="No Such Sheet"):
            RawDataLoader(replace(loader_settings, sheets=sheets)).load(use_cache=False)

    def test_missing_file_raises(self, loader_settings: LoaderSettings, tmp_path: Path) -> None:
        """Absent workbook raises MissingFileError pointing to data-check."""
        loader = RawDataLoader(replace(loader_settings, raw_path=tmp_path / "absent.xlsx"))
        with pytest.raises(MissingFileError, match="data-check"):
            loader.load(use_cache=False)

    def test_sheet_names(self, loader_settings: LoaderSettings) -> None:
        """Lists the workbook's sheets."""
        names = RawDataLoader(loader_settings).sheet_names()
        assert set(loader_settings.sheets.values()) == set(names)


class TestLoaderSettings:
    """LoaderSettings.from_config takes paths and mappings from config."""

    def test_from_config(self, cfg: Config, tmp_path: Path) -> None:
        """Raw path is under data/raw; cache dir is the one passed in."""
        settings = LoaderSettings.from_config(cfg, cache_dir=tmp_path)
        assert settings.raw_path.parent == DATA_RAW
        assert settings.cache_dir == tmp_path
        assert set(settings.sheets) == set(Sheet)
