"""Download LOBSTER sample files (2012-06-21) from Hugging Face `totalorganfailure/lobster-data`.

Raw CSVs are fetched directly via `resolve/main` URLs; `datasets.load_dataset` is known to be
broken for this repo and is deliberately not used.

Usage:
    python data/download.py                 # AAPL, MSFT, GOOG, INTC, AMZN at 10 levels
    python data/download.py --spy           # also SPY at 30 levels (~940 MB, 09:30-10:30 only)
"""
import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/totalorganfailure/lobster-data/resolve/main"
DATE = "2012-06-21"
RAW_DIR = Path(__file__).resolve().parent / "raw"
TRAIN_TICKERS = ["AAPL", "MSFT", "GOOG", "INTC", "AMZN"]
# SPY's sample file only spans 34200000-37800000 ms (09:30-10:30), unlike the other tickers.
WINDOW = {"SPY": "34200000_37800000"}
DEFAULT_WINDOW = "34200000_57600000"


def file_names(ticker: str, levels: int) -> list[str]:
    window = WINDOW.get(ticker, DEFAULT_WINDOW)
    folder = f"LOBSTER_SampleFile_{ticker}_{DATE}_{levels}"
    return [f"{folder}/{ticker}_{DATE}_{window}_{kind}_{levels}.csv" for kind in ("message", "orderbook")]


def download(rel_path: str) -> Path:
    dest = RAW_DIR / Path(rel_path).name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip  {dest.name} (exists)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"fetch {rel_path}", flush=True)
    with urllib.request.urlopen(f"{BASE}/{rel_path}") as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.replace(dest)
    print(f"done  {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spy", action="store_true", help="also download SPY 30-level files")
    parser.add_argument("--tickers", nargs="*", default=TRAIN_TICKERS)
    args = parser.parse_args(argv)

    jobs = [(t, 10) for t in args.tickers]
    if args.spy:
        jobs.append(("SPY", 30))
    for ticker, levels in jobs:
        for rel in file_names(ticker, levels):
            download(rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
