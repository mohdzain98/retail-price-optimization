"""Raw workbook loading for the Breakfast at the Frat dataset.

RawDataLoader reads the three sheets, auto-detects the header row, renames
columns to canonical `Col` names and caches the tables as parquet so Excel is
parsed only once. All paths and mappings arrive via LoaderSettings.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd

from src.config import Config, Sheet
from src.schema import Col
from src.utils import AppLogger, DataValidationError, MissingFileError

logger = AppLogger.get_logger(__name__)


@dataclass(frozen=True)
class RawData:
    """The three raw tables with canonical column names.

    Transactions are store x UPC x week; products and stores are lookups.
    """

    transactions: pd.DataFrame
    products: pd.DataFrame
    stores: pd.DataFrame

    @classmethod
    def from_tables(cls, tables: Mapping[Sheet, pd.DataFrame]) -> RawData:
        """Build from a Sheet -> DataFrame mapping."""
        return cls(
            transactions=tables[Sheet.TRANSACTIONS],
            products=tables[Sheet.PRODUCTS],
            stores=tables[Sheet.STORES],
        )


@dataclass(frozen=True)
class LoaderSettings:
    """Everything RawDataLoader needs: paths, sheet names and column mappings.

    `header_row` None means auto-detect by scanning for the first mapped column.
    """

    raw_path: Path
    cache_dir: Path
    sheets: Mapping[Sheet, str]
    columns: Mapping[Sheet, Mapping[str, Col]]
    header_row: int | None = None

    @classmethod
    def from_config(cls, cfg: Config, cache_dir: Path) -> LoaderSettings:
        """Settings from the data section of the config."""
        return cls(
            raw_path=cfg.data.raw_path,
            cache_dir=cache_dir,
            sheets=cfg.data.sheets,
            columns=cfg.data.columns,
            header_row=cfg.data.header_row,
        )


class RawDataLoader:
    """Reads the raw workbook into RawData, with a per-sheet parquet cache.

    Fails loudly (DataValidationError) when a sheet or mapped column is missing.
    """

    HEADER_SCAN_ROWS: ClassVar[int] = 10
    CACHE_PREFIX: ClassVar[str] = "raw_"

    def __init__(self, settings: LoaderSettings) -> None:
        self._settings = settings

    def load(self, use_cache: bool = True) -> RawData:
        """Return the raw tables, from cache when available.

        A cache miss reads Excel (slow) and rewrites the cache.
        """
        if use_cache and all(self._cache_path(s).exists() for s in Sheet):
            logger.info("Loading raw tables from parquet cache")
            return RawData.from_tables({s: pd.read_parquet(self._cache_path(s)) for s in Sheet})
        self._require_file()
        tables = {sheet: self._read_sheet(sheet) for sheet in Sheet}
        tx = tables[Sheet.TRANSACTIONS]
        tables[Sheet.TRANSACTIONS] = tx.assign(**{Col.WEEK: pd.to_datetime(tx[Col.WEEK])})
        self._write_cache(tables)
        return RawData.from_tables(tables)

    def sheet_names(self) -> list[str]:
        """Sheet names present in the workbook (raises if the file is missing)."""
        self._require_file()
        return [str(name) for name in pd.ExcelFile(self._settings.raw_path).sheet_names]

    def _require_file(self) -> None:
        """Raise MissingFileError with download guidance if the workbook is absent."""
        if not self._settings.raw_path.exists():
            raise MissingFileError(
                f"{self._settings.raw_path} not found; run `make data-check` for download steps"
            )

    def _read_sheet(self, sheet: Sheet) -> pd.DataFrame:
        """Read one sheet, keep mapped columns and rename them to canonical names."""
        name, mapping = self._settings.sheets[sheet], dict(self._settings.columns[sheet])
        header = self._settings.header_row
        if header is None:
            header = self._detect_header(name, next(iter(mapping)))
        logger.info("Reading sheet", extra={"sheet": name, "header_row": header})
        frame = self._excel(name, header=header)
        frame.columns = pd.Index([str(c).strip().upper() for c in frame.columns])
        missing = [c for c in mapping if c not in frame.columns]
        if missing:
            raise DataValidationError(
                f"Sheet '{name}' is missing columns {missing}; found {list(frame.columns)}"
            )
        return frame[list(mapping)].rename(columns=mapping)

    def _detect_header(self, sheet_name: str, first_col: str) -> int:
        """0-based row whose cells contain first_col (case-insensitive)."""
        head = self._excel(sheet_name, header=None, nrows=self.HEADER_SCAN_ROWS)
        normalized = head.astype(str).apply(lambda col: col.str.strip().str.upper())
        hits = normalized.eq(first_col.upper()).any(axis=1)
        if not hits.any():
            raise DataValidationError(f"No header with '{first_col}' in sheet '{sheet_name}'")
        return int(hits.idxmax())

    def _excel(self, sheet_name: str, header: int | None, nrows: int | None = None) -> pd.DataFrame:
        """pd.read_excel wrapper that turns a missing sheet into DataValidationError."""
        try:
            return pd.read_excel(
                self._settings.raw_path, sheet_name=sheet_name, header=header, nrows=nrows
            )
        except ValueError as err:
            raise DataValidationError(f"Cannot read sheet '{sheet_name}': {err}") from err

    def _cache_path(self, sheet: Sheet) -> Path:
        """Parquet cache location for one sheet."""
        return self._settings.cache_dir / f"{self.CACHE_PREFIX}{sheet.value}.parquet"

    def _write_cache(self, tables: Mapping[Sheet, pd.DataFrame]) -> None:
        """Persist every table to its parquet cache file."""
        self._settings.cache_dir.mkdir(parents=True, exist_ok=True)
        for sheet, frame in tables.items():
            frame.to_parquet(self._cache_path(sheet), index=False)
