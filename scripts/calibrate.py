"""Measure per-ticker scale statistics and calibrate the price-impact coefficient.

Writes configs/calibration.json. Every number in the README's calibration table comes from here.
Usage: python scripts/calibrate.py [--form linear|sqrt]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.lobster_data import load_day  # noqa: E402
from env.normalization import encode_book, reference_stats  # noqa: E402
from env.price_impact import calibrate_lambda, impact_ramp  # noqa: E402

TICKERS = ["AAPL", "MSFT", "GOOG", "INTC", "AMZN"]
TRAIN_TICKERS = ["AAPL", "MSFT", "GOOG", "INTC"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--form", default="linear", choices=["linear", "sqrt"])
    args = parser.parse_args()

    out = {"form": args.form, "tickers": {}}
    header = f"{'ticker':6} {'events':>9} {'mid$':>8} {'spread$':>8} {'spr_ticks':>9} {'touch_dep':>9} " \
             f"{'p95_new':>8} {'med_move$':>9} {'lambda':>8} {'|p1| med':>8} {'s1 med':>7}"
    print(header)
    for t in TICKERS:
        d = load_day(t)
        s = reference_stats(d)
        lam, tg = calibrate_lambda(d, s, args.form)
        feats = encode_book(d, s)
        row = {
            "events": len(d),
            "median_mid": float(np.nanmedian(d.mid)),
            "median_spread_ticks": float(np.nanmedian(d.spread) / 0.01),
            **tg,
            "lambda": lam,
            "median_abs_best_ask_price_feature": float(np.median(np.abs(feats[:, 0]))),
            "median_best_ask_size_feature": float(np.median(feats[:, 1])),
        }
        out["tickers"][t] = row
        print(f"{t:6} {len(d):9d} {row['median_mid']:8.2f} {tg['median_ref_spread']:8.4f} "
              f"{row['median_spread_ticks']:9.1f} {tg['median_touch_depth']:9.1f} {tg['p95_new_order_size']:8.0f} "
              f"{tg['median_nonzero_mid_move']:9.4f} {lam:8.4f} {row['median_abs_best_ask_price_feature']:8.3f} "
              f"{row['median_best_ask_size_feature']:7.3f}")

    print("impact ramp beta(W)/beta(200), W = events:")
    for t in TICKERS:
        d = load_day(t)
        ramp = impact_ramp(d, reference_stats(d), args.form)
        out["tickers"][t]["ramp"] = {str(w): r for w, r in ramp.items()}
        print(f"  {t}: " + "  ".join(f"W={w}: {r:.2f}" for w, r in ramp.items()))

    # Pooled lambda is reported for reference only; the environment uses per-ticker values.
    out["pooled_lambda_train"] = float(np.median([out["tickers"][t]["lambda"] for t in TRAIN_TICKERS]))
    print(f"pooled lambda (median over {TRAIN_TICKERS}): {out['pooled_lambda_train']:.4f}")

    path = ROOT / "configs" / "calibration.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
