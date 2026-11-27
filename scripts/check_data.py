"""Check that the raw Breakfast at the Frat workbook is in place.

The dataset sits behind a free form on dunnhumby's site, so it cannot be
downloaded automatically. RawDataCheck reports whether the file exists,
compares its sheets with config and logs download steps when it is missing.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping

from src.config import DATA_PROCESSED, DATA_RAW, Config, Sheet
from src.prep import LoaderSettings, RawDataLoader
from src.utils import AppLogger, MissingFileError

logger = AppLogger.get_logger(__name__)

DOWNLOAD_STEPS = """Raw dataset not found. To get it:
  1. Open https://www.dunnhumby.com/source-files/
  2. Choose "Breakfast at the Frat", fill in the short form, download the zip.
  3. Extract the .xlsx into: {raw_dir}
  4. If the filename differs, update `data.raw_file` in config/config.yaml.
  5. Re-run: make data-check"""


class RawDataCheck:
    """Verifies the raw workbook exists and has the sheets named in config.

    Exit codes: 0 ready, 1 file missing, 2 sheet names do not match.
    """

    def __init__(self, loader: RawDataLoader, expected_sheets: Mapping[Sheet, str]) -> None:
        self._loader = loader
        self._expected = expected_sheets

    def run(self) -> int:
        """Log the dataset status and return an exit code."""
        try:
            found = set(self._loader.sheet_names())
        except MissingFileError:
            logger.warning(DOWNLOAD_STEPS.format(raw_dir=DATA_RAW))
            return 1
        missing = {s.value: n for s, n in self._expected.items() if n not in found}
        if missing:
            logger.error(
                "Config sheet names not in workbook; update `data.sheets` in config.yaml",
                extra={"missing": missing, "found": sorted(found)},
            )
            return 2
        logger.info("Data ready", extra={"sheets": sorted(found)})
        return 0


if __name__ == "__main__":
    AppLogger.init()
    cfg = Config.load()
    settings = LoaderSettings.from_config(cfg, cache_dir=DATA_PROCESSED)
    sys.exit(RawDataCheck(RawDataLoader(settings), cfg.data.sheets).run())
