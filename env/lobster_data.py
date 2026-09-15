"""Load LOBSTER message/orderbook CSV pairs into numpy arrays.

File format (no header rows):
  message:   time(s after midnight), event_type, order_id, size, price(x10^4), direction
  orderbook: per level 1..L -> ask_price, ask_size, bid_price, bid_size  (price x10^4)
Row N of the orderbook is the book state immediately after message event N.

Empty book levels use LOBSTER dummy prices (+9999999999 ask / -9999999999 bid, size 0);
they are converted to NaN price / 0 size here.

Parsed arrays are cached as .npy under data/cache/ because the CSVs take tens of seconds to parse.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"
DATE = "2012-06-21"
WINDOWS = {"SPY": "34200000_37800000"}
DEFAULT_WINDOW = "34200000_57600000"
PRICE_SCALE = 10_000.0
DUMMY_PRICE = 9_999_999_999

# Event type codes
NEW, PARTIAL_CANCEL, DELETE, EXEC_VISIBLE, EXEC_HIDDEN, HALT = 1, 2, 3, 4, 5, 7


@dataclass
class LobsterDay:
    ticker: str
    levels: int
    time: np.ndarray        # (N,) float64 seconds after midnight
    event_type: np.ndarray  # (N,) int8
    order_id: np.ndarray    # (N,) int64
    size: np.ndarray        # (N,) int64 shares
    price: np.ndarray       # (N,) float64 dollars
    direction: np.ndarray   # (N,) int8, 1 buy / -1 sell
    ask_price: np.ndarray   # (N, L) float64 dollars, NaN for empty level
    ask_size: np.ndarray    # (N, L) float64 shares
    bid_price: np.ndarray   # (N, L)
    bid_size: np.ndarray    # (N, L)

    def __len__(self) -> int:
        return len(self.time)

    @property
    def mid(self) -> np.ndarray:
        return (self.ask_price[:, 0] + self.bid_price[:, 0]) / 2.0

    @property
    def spread(self) -> np.ndarray:
        return self.ask_price[:, 0] - self.bid_price[:, 0]

    def truncate_levels(self, levels: int) -> "LobsterDay":
        """Return a view keeping only the top `levels` book levels (e.g. SPY 30 -> 10)."""
        return LobsterDay(self.ticker, levels, self.time, self.event_type, self.order_id,
                          self.size, self.price, self.direction,
                          self.ask_price[:, :levels], self.ask_size[:, :levels],
                          self.bid_price[:, :levels], self.bid_size[:, :levels])


def csv_paths(ticker: str, levels: int, raw_dir: Path = RAW_DIR) -> tuple[Path, Path]:
    window = WINDOWS.get(ticker, DEFAULT_WINDOW)
    stem = f"{ticker}_{DATE}_{window}"
    return raw_dir / f"{stem}_message_{levels}.csv", raw_dir / f"{stem}_orderbook_{levels}.csv"


def _parse(ticker: str, levels: int, raw_dir: Path) -> dict[str, np.ndarray]:
    msg_path, ob_path = csv_paths(ticker, levels, raw_dir)
    if not msg_path.exists() or not ob_path.exists():
        raise FileNotFoundError(f"missing LOBSTER files for {ticker}; run `python data/download.py`")
    msg = pd.read_csv(msg_path, header=None).to_numpy()
    if msg.shape[1] != 6:
        raise ValueError(f"{msg_path.name}: expected 6 columns, got {msg.shape[1]}")
    ob = pd.read_csv(ob_path, header=None, dtype=np.int64).to_numpy()
    if ob.shape[1] != 4 * levels:
        raise ValueError(f"{ob_path.name}: expected {4 * levels} columns, got {ob.shape[1]}")
    if len(msg) != len(ob):
        raise ValueError(f"{ticker}: message rows {len(msg)} != orderbook rows {len(ob)}")
    return {
        "time": msg[:, 0].astype(np.float64),
        "event_type": msg[:, 1].astype(np.int8),
        "order_id": msg[:, 2].astype(np.int64),
        "size": msg[:, 3].astype(np.int64),
        "price_raw": msg[:, 4].astype(np.int64),
        "direction": msg[:, 5].astype(np.int8),
        "book_raw": ob,
    }


def load_day(ticker: str, levels: int = 10, raw_dir: Path = RAW_DIR,
             cache_dir: Path | None = CACHE_DIR) -> LobsterDay:
    arrays: dict[str, np.ndarray] | None = None
    names = ("time", "event_type", "order_id", "size", "price_raw", "direction", "book_raw")
    if cache_dir is not None:
        cdir = cache_dir / f"{ticker}_{levels}"
        if all((cdir / f"{n}.npy").exists() for n in names):
            arrays = {n: np.load(cdir / f"{n}.npy") for n in names}
    if arrays is None:
        arrays = _parse(ticker, levels, raw_dir)
        if cache_dir is not None:
            cdir.mkdir(parents=True, exist_ok=True)
            for n in names:
                np.save(cdir / f"{n}.npy", arrays[n])

    book = arrays["book_raw"].reshape(-1, levels, 4)
    ask_p, ask_s, bid_p, bid_s = (book[:, :, i] for i in range(4))
    ask_empty = np.abs(ask_p) >= DUMMY_PRICE
    bid_empty = np.abs(bid_p) >= DUMMY_PRICE
    return LobsterDay(
        ticker=ticker,
        levels=levels,
        time=arrays["time"],
        event_type=arrays["event_type"],
        order_id=arrays["order_id"],
        size=arrays["size"],
        price=arrays["price_raw"] / PRICE_SCALE,
        direction=arrays["direction"],
        ask_price=np.where(ask_empty, np.nan, ask_p / PRICE_SCALE),
        ask_size=np.where(ask_empty, 0, ask_s).astype(np.float64),
        bid_price=np.where(bid_empty, np.nan, bid_p / PRICE_SCALE),
        bid_size=np.where(bid_empty, 0, bid_s).astype(np.float64),
    )
