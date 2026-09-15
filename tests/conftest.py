import sys
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.lobster_data import csv_paths, load_day  # noqa: E402


@lru_cache(maxsize=None)
def _day(ticker: str, levels: int = 10):
    return load_day(ticker, levels)


@pytest.fixture
def day():
    """Factory fixture: day("AAPL") -> LobsterDay, skipping if the raw files are absent."""
    def get(ticker: str, levels: int = 10):
        msg, ob = csv_paths(ticker, levels)
        if not (msg.exists() and ob.exists()):
            pytest.skip(f"{ticker} data not downloaded (python data/download.py)")
        return _day(ticker, levels)
    return get
