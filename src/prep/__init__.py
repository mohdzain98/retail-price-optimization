"""Data-prep layer (Phase 1): raw workbook -> modelling panel.

loader.py reads the workbook, panel.py cleans/aggregates/filters, features.py
adds features and the split, trend.py adds the LOESS trend, service.py is the
facade. Other layers import from here: from src.prep import DataPrepFactory.
"""

from src.prep.loader import LoaderSettings, RawData, RawDataLoader
from src.prep.service import DataPrepFactory, DataPrepService

__all__ = ["DataPrepFactory", "DataPrepService", "LoaderSettings", "RawData", "RawDataLoader"]
