"""Check that the raw Breakfast at the Frat workbook is in place.

The dataset sits behind a free form on dunnhumby's site, so it cannot be
downloaded automatically. This script reports whether the file exists, lists
its sheets against config, and logs download steps when it is missing.
"""

from __future__ import annotations

import sys

from src.config import DATA_RAW, load_config
from src.utils.logger import AppLogger

AppLogger.init()
logger = AppLogger.get_logger(__name__)

DOWNLOAD_STEPS = """Raw dataset not found. To get it:
  1. Open dunnhumby's "Source Files" page (search: "dunnhumby source files").
  2. Choose "Breakfast at the Frat", fill in the short form, download the zip.
  3. Extract the .xlsx into: {raw_dir}
  4. If the filename differs, update `data.raw_file` in config/config.yaml.
  5. Re-run: make data-check"""


def main() -> int:
    """Report dataset status; return a non-zero exit code when not ready.

    Opens the workbook read-only to compare its sheet names with config.
    """
    cfg = load_config()
    path = cfg.data.raw_path
    if not path.exists():
        logger.warning(DOWNLOAD_STEPS.format(raw_dir=DATA_RAW))
        return 1

    from openpyxl import load_workbook

    found = set(load_workbook(path, read_only=True).sheetnames)
    expected = cfg.data.sheets.model_dump()
    missing = {k: v for k, v in expected.items() if v not in found}

    logger.info("Found %s", path.name, extra={"size_mb": round(path.stat().st_size / 1e6, 1)})
    logger.print("Sheets in workbook", sorted(found))
    if missing:
        logger.error(
            "Config sheet names not in workbook; update `data.sheets` in config.yaml",
            extra={"missing": missing, "found": sorted(found)},
        )
        return 2
    logger.info("Data ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
