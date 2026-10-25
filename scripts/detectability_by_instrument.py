"""Why does attribution-free detection work on INTC and MSFT but not on AAPL, GOOG and AMZN?

`results/anonymised_experiment.md` shows a sharp per-instrument split: with the participant block withheld,
logistic regression reaches step-level F1 0.585 on SCRIPTED-ATK_INTC and 0.600 on MSFT, but 0.073-0.131 on
AAPL, GOOG and AMZN, where it loses to a constant classifier.

The scripted attacker is the same program on every stock, so the difference is a property of the market, not
of the attack. This script tests two obvious mechanical explanations. **Both are refuted**, and that is the
result: the per-instrument split is real and reproducible, but it is not explained by any single order-book
statistic measured here.

H1  The spoof is proportionally larger on the detectable stocks.
    Refuted: the attacker places 10.00x the touch depth on all five, by construction.

H2  The spoof moves the displayed book more on the detectable stocks, or moves it further relative to the
    book's natural variation (signal-to-noise).
    Refuted: the shift in normalised deep imbalance is *smaller* on INTC and MSFT than on AAPL and AMZN, and
    the signal-to-noise ratio does not rank with detectability either.

What remains is a descriptive association with spread width, on five instruments. That is far too small a
sample to support a mechanism, and this script does not claim one.

Usage:
    python scripts/detectability_by_instrument.py [--data data/watchdog_v2]

Writes results/detectability_by_instrument.json and .md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from watchdog.anonymised_experiment import book_features  # noqa: E402
from watchdog.dataset import load_split, orders_from_array  # noqa: E402

DEEP_IMBALANCE = 3  # column within book_features()


def per_instrument_f1(results: Path) -> dict:
    """Step-level F1 per source for the best attribution-free model, read from the experiment's output."""
    p = results / "anonymised_experiment.json"
    if not p.exists():
        return {}
    pl = json.loads(p.read_text(encoding="utf-8"))
    held = pl["headline"]["held_out_set"]
    model = pl["headline"]["best_step_model"]
    per_seed = pl.get("per_seed", [])
    if not per_seed or model not in per_seed[0]:
        return {}
    by_src = per_seed[0][model]["sets"][held].get("by_source", {})
    return {k.split("_")[-1]: v["step"]["f1"] for k, v in by_src.items() if k.startswith("SCRIPTED")}


def measure(data: Path, cal: dict, f1: dict) -> list[dict]:
    held = load_split("heldout", data)
    rows = []
    for name in sorted(held):
        if not name.startswith("SCRIPTED"):
            continue
        tk = name.split("_")[1]
        s = held[name]
        orders = orders_from_array(s["orders"])
        mult = np.array([o.depth_mult for _, o in orders if o.manipulative], dtype=float)
        imb = book_features(s["X"])[:, DEEP_IMBALANCE]
        y = s["y"].astype(bool)
        shift = float(abs(imb[y].mean() - imb[~y].mean())) if y.any() and (~y).any() else float("nan")
        noise = float(imb[~y].std())
        rows.append({
            "ticker": tk,
            "median_spread_ticks": cal[tk]["median_spread_ticks"],
            "median_touch_depth": cal[tk]["median_touch_depth"],
            "spoof_size_over_depth": float(mult.mean()) if mult.size else float("nan"),
            "book_imbalance_shift": shift,
            "book_imbalance_noise": noise,
            "signal_to_noise": shift / noise if noise > 0 else float("nan"),
            "step_f1": f1.get(tk, float("nan")),
        })
    return rows


def corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def render(rows: list[dict], cors: dict) -> str:
    L = ["# Why detection works on some instruments and not others", "",
         "With participant attribution withheld, step-level detection splits sharply by instrument. The "
         "scripted attacker is the same program on every stock, so the difference belongs to the market. Two "
         "mechanical explanations were tested. **Both are refuted.**", "",
         "| Stock | Spread (ticks) | Touch depth | Spoof / depth | Book shift | Book noise (sd) | Signal/noise | Step F1 |",
         "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["step_f1"] if np.isfinite(r["step_f1"]) else 1):
        L.append(f"| {r['ticker']} | {r['median_spread_ticks']:.0f} | {r['median_touch_depth']:,.0f} | "
                 f"{_f(r['spoof_size_over_depth'], 2)} | {_f(r['book_imbalance_shift'], 4)} | "
                 f"{_f(r['book_imbalance_noise'], 4)} | {_f(r['signal_to_noise'])} | "
                 f"**{_f(r['step_f1'])}** |")
    L += ["", "## H1 — the spoof is proportionally larger where it is detected. **Refuted.**", "",
          "The attacker places the same multiple of touch depth on every stock "
          f"({_f(rows[0]['spoof_size_over_depth'], 2)}x, by construction), so attack size cannot explain any "
          "difference.", "",
          "## H2 — the spoof moves the book more, or further relative to its noise. **Refuted.**", "",
          f"Correlation of the book-imbalance shift with step F1: **{_f(cors['shift'])}**. "
          f"Correlation of signal-to-noise with step F1: **{_f(cors['snr'])}**. Neither explains the split, and "
          "the shift is in fact *smaller* on the two detectable stocks than on AAPL and AMZN. The "
          "signal-to-noise ranking disagrees with the detectability ranking outright.", "",
          "## What is left", "",
          f"A descriptive association with spread width (correlation **{_f(cors['spread'])}** with step F1) on "
          "**five instruments**. That is far too small a sample to support a mechanism, and none is claimed "
          "here. The per-instrument split is a reproducible observation without an established cause.", "",
          "Two further possibilities were *not* tested and would need more instruments or more trading days: "
          "that the effect is multivariate across the 153 features rather than present in any single "
          "order-book statistic, and that it tracks tick-size regime (both detectable stocks are 1-tick) "
          "rather than spread width as such — which this dataset cannot separate, because the two coincide.", ""]
    return "\n".join(L) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data" / "watchdog_v2"))
    p.add_argument("--out", default=str(ROOT / "results"))
    args = p.parse_args()

    data, out = Path(args.data), Path(args.out)
    if not (data / "heldout").exists():
        print(f"no recorded held-out data under {data}; run watchdog/dataset.py first")
        return 1
    cal_path = ROOT / "configs" / "calibration.json"
    if not cal_path.exists():
        print("no configs/calibration.json; run scripts/calibrate.py first")
        return 1
    cal = json.loads(cal_path.read_text(encoding="utf-8"))["tickers"]

    f1 = per_instrument_f1(out)
    if not f1:
        print("no per-source results; run watchdog/anonymised_experiment.py first")
        return 1
    rows = measure(data, cal, f1)
    cors = {"shift": corr([r["book_imbalance_shift"] for r in rows], [r["step_f1"] for r in rows]),
            "snr": corr([r["signal_to_noise"] for r in rows], [r["step_f1"] for r in rows]),
            "spread": corr([r["median_spread_ticks"] for r in rows], [r["step_f1"] for r in rows])}

    payload = {"data": str(data), "rows": rows, "correlations": cors, "n_instruments": len(rows),
               "verdict": "H1 refuted (attack size identical); H2 refuted (shift and SNR do not rank with "
                          "detectability); no mechanism established at n=5"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "detectability_by_instrument.json").write_text(json.dumps(payload, indent=2, default=float),
                                                          encoding="utf-8")
    (out / "detectability_by_instrument.md").write_text(render(rows, cors), encoding="utf-8")
    for k, v in cors.items():
        print(f"  corr({k}, step F1) = {v: .3f}")
    print(f"wrote {out / 'detectability_by_instrument.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
