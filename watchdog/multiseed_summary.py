"""Headline Watchdog metrics across training seeds, with 95% confidence intervals.

Usage:
    python watchdog/multiseed_summary.py [--tags main main_s2 main_s3]

Every seed is scored on the same recorded data: the test and heldout splits (buckets as in
watchdog/evaluate_watchdog.py) and the saved LATEBURST-ATK data from the pre-registered test. The rule baseline
is seed-independent and reported once. The pre-registered result itself stays the single run on the `main`
Watchdog; the other seeds here only measure how much the headline numbers vary with the training seed.

Writes results/multiseed.md and results/multiseed.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.baseline_detector import tune  # noqa: E402
from watchdog.dataset import DATA_DIR, load_split, orders_from_array  # noqa: E402
from watchdog.evaluate_watchdog import BUCKETS, combine, score_source  # noqa: E402
from watchdog.watchdog_env import load_obs_norm  # noqa: E402

# two-sided 95% Student-t critical values by sample size
T_975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def mean_ci95(values) -> tuple[float, float]:
    """(mean, half-width) of a Student-t 95% CI; half-width NaN for fewer than 2 finite values."""
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) < 2:
        return float(x.mean()), float("nan")
    t = T_975.get(len(x), 1.96)
    return float(x.mean()), float(t * x.std(ddof=1) / math.sqrt(len(x)))


def headline(bucket_rows: dict, lateburst: dict) -> dict:
    ind = bucket_rows["In-distribution (test split)"]
    return {
        "in-distribution order F1": ind["watchdog_orders"]["f1"],
        "in-distribution order precision": ind["watchdog_orders"]["precision"],
        "in-distribution order recall": ind["watchdog_orders"]["recall"],
        "in-distribution step precision": ind["watchdog_steps"]["precision"],
        "in-distribution step recall": ind["watchdog_steps"]["recall"],
        "legitimate-only order FPR": bucket_rows["Legitimate only (HONEST + FLICKER)"]["watchdog_orders"]["fpr"],
        "held-out AMZN legitimate order FPR": bucket_rows["Held-out RL: SPOOFER-05 + AMZN controls"]["watchdog_orders"]["fpr"],
        "LATEBURST-ATK order recall": lateburst["watchdog_orders"]["recall"],
        "LATEBURST-ATK order precision": lateburst["watchdog_orders"]["precision"],
        "LATEBURST-ATK order FPR": lateburst["watchdog_orders"]["fpr"],
    }


def rule_headline(bucket_rows: dict, lateburst: dict) -> dict:
    ind = bucket_rows["In-distribution (test split)"]
    return {
        "in-distribution order F1": ind["rule_orders"]["f1"],
        "in-distribution order precision": ind["rule_orders"]["precision"],
        "in-distribution order recall": ind["rule_orders"]["recall"],
        "legitimate-only order FPR": bucket_rows["Legitimate only (HONEST + FLICKER)"]["rule_orders"]["fpr"],
        "held-out AMZN legitimate order FPR": bucket_rows["Held-out RL: SPOOFER-05 + AMZN controls"]["rule_orders"]["fpr"],
        "LATEBURST-ATK order recall": lateburst["rule_orders"]["recall"],
        "LATEBURST-ATK order precision": lateburst["rule_orders"]["precision"],
        "LATEBURST-ATK order FPR": lateburst["rule_orders"]["fpr"],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="*", default=["main", "main_s2", "main_s3"])
    args = p.parse_args()

    from sb3_contrib import RecurrentPPO

    train, test, held = load_split("train"), load_split("test"), load_split("heldout")
    late = load_split("lateburst")
    if not (train and test and held and late):
        print("missing data: need train/test/heldout splits and data/watchdog/lateburst (run lateburst_test.py)")
        return 1
    det, _ = tune([o for s in train.values() for _, o in orders_from_array(s["orders"])])

    t0 = time.time()
    per_seed, rule = {}, None
    for tag in args.tags:
        path = ROOT / "checkpoints" / "WATCHDOG" / tag / "model.zip"
        if not path.exists():
            print(f"[skip] {tag}: no model at {path.relative_to(ROOT)}")
            continue
        model = RecurrentPPO.load(path, device="cpu")
        obs_norm = load_obs_norm(path.parent)
        scored = {(sp, n): score_source(model, src, det, obs_norm)
                  for sp, group in (("test", test), ("heldout", held)) for n, src in group.items()}
        bucket_rows = {name: combine([v for k, v in scored.items() if pred(*k)]) for name, pred in BUCKETS}
        late_row = combine([score_source(model, src, det, obs_norm) for src in late.values()])
        per_seed[tag] = headline(bucket_rows, late_row)
        rule = rule or rule_headline(bucket_rows, late_row)
        print(f"[{time.time() - t0:5.0f}s] scored {tag}", flush=True)
    if not per_seed:
        print("no Watchdog models found")
        return 1

    metrics = list(next(iter(per_seed.values())))
    summary = {m: dict(zip(("mean", "ci95"), mean_ci95([per_seed[t][m] for t in per_seed]))) for m in metrics}
    payload = {"tags": list(per_seed), "per_seed": per_seed, "summary": summary, "rule": rule,
               "rule_params": {"n_mult": det.n_mult, "m_events": det.m_events}, "runtime_s": round(time.time() - t0, 1)}
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "multiseed.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out / "multiseed.md").write_text(render(payload), encoding="utf-8")
    print(f"wrote {out / 'multiseed.md'}")
    return 0


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def render(pl: dict) -> str:
    tags = pl["tags"]
    L = ["# Watchdog across training seeds", "",
         f"Seeds (checkpoint tags): {', '.join(tags)}. 95% CI uses Student-t with n = {len(tags)}. Every seed is scored "
         "on the same recorded data. The rule baseline does not depend on the seed.", "",
         "| Metric | " + " | ".join(tags) + " | Mean ± 95% CI | Rule |",
         "|---|" + "---|" * len(tags) + "---|---|"]
    for m, s in pl["summary"].items():
        vals = " | ".join(_f(pl["per_seed"][t][m]) for t in tags)
        rule = _f(pl["rule"].get(m)) if pl["rule"] and m in pl["rule"] else "—"
        L.append(f"| {m} | {vals} | {_f(s['mean'])} ± {_f(s['ci95'])} | {rule} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
