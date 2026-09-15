"""Which features does the trained Watchdog rely on? Feature-group ablation on the test split.

Usage:
    python watchdog/feature_ablation.py [--tag main]

Each group of the 46-dim observation is neutralised by replacing it with its training mean (the saved
VecNormalize mean, which becomes exactly 0 after normalisation, so no out-of-range values are introduced).
The same test-split data is then rescored. A large drop in F1 / recall or a rise in false positives means the
Watchdog depends on that group. No retraining; the model is unchanged.

Writes results/feature_ablation.md and results/feature_ablation.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.baseline_detector import tune  # noqa: E402
from evaluation.rollout import BOOK_DIM, OBS_DIM  # noqa: E402
from watchdog.dataset import load_split, orders_from_array  # noqa: E402
from watchdog.evaluate_watchdog import combine, score_source  # noqa: E402
from watchdog.watchdog_env import load_obs_norm  # noqa: E402

LEVELS = BOOK_DIM // 4
# book features per level are [ask price, ask size, bid price, bid size]
GROUPS: dict[str, list[int]] = {
    "book price offsets (all levels)": [4 * lv + j for lv in range(LEVELS) for j in (0, 2)],
    "book sizes (all levels)": [4 * lv + j for lv in range(LEVELS) for j in (1, 3)],
    "best level only (4)": [0, 1, 2, 3],
    "all book features (40)": list(range(BOOK_DIM)),
    "participant: resting size": [BOOK_DIM + 0],
    "participant: oldest order age": [BOOK_DIM + 1],
    "participant: placed this step": [BOOK_DIM + 2],
    "participant: cancelled this step": [BOOK_DIM + 3],
    "participant: trade direction": [BOOK_DIM + 4],
    "participant: resting side imbalance": [BOOK_DIM + 5],
    "all participant features (6)": list(range(BOOK_DIM, OBS_DIM)),
}


def ablate(X: np.ndarray, idx: list[int], fill: np.ndarray) -> np.ndarray:
    """Copy of X with columns `idx` replaced by `fill[idx]`; other columns untouched."""
    out = X.copy()
    out[:, idx] = fill[idx]
    return out


def headline(scored: dict) -> dict:
    ind = combine(list(scored.values()))
    legit = combine([v for k, v in scored.items() if k.startswith(("HONEST", "FLICKER"))])
    return {
        "order F1": ind["watchdog_orders"]["f1"],
        "order recall": ind["watchdog_orders"]["recall"],
        "order precision": ind["watchdog_orders"]["precision"],
        "step recall": ind["watchdog_steps"]["recall"],
        "step precision": ind["watchdog_steps"]["precision"],
        "legitimate order FPR": legit["watchdog_orders"]["fpr"],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="main")
    args = p.parse_args()

    from sb3_contrib import RecurrentPPO

    path = ROOT / "checkpoints" / "WATCHDOG" / args.tag / "model.zip"
    if not path.exists():
        print(f"no Watchdog at {path}")
        return 1
    model = RecurrentPPO.load(path, device="cpu")
    obs_norm = load_obs_norm(path.parent)
    if obs_norm is None:
        print("this ablation needs the saved observation normalisation (train with --norm-obs)")
        return 1
    train, test = load_split("train"), load_split("test")
    if not (train and test):
        print("missing train/test splits; run watchdog/dataset.py first")
        return 1
    det, _ = tune([o for s in train.values() for _, o in orders_from_array(s["orders"])])
    fill = obs_norm["mean"]

    t0 = time.time()
    results = {"none": headline({n: score_source(model, s, det, obs_norm) for n, s in test.items()})}
    print(f"[{time.time() - t0:5.0f}s] baseline: " + ", ".join(f"{k}={v:.3f}" for k, v in results["none"].items()),
          flush=True)
    for group, idx in GROUPS.items():
        scored = {n: score_source(model, {**s, "X": ablate(s["X"], idx, fill)}, det, obs_norm) for n, s in test.items()}
        results[group] = headline(scored)
        print(f"[{time.time() - t0:5.0f}s] {group}: " + ", ".join(f"{k}={v:.3f}" for k, v in results[group].items()),
              flush=True)

    payload = {"watchdog": str(path.relative_to(ROOT)), "fill": "training mean (0 after normalisation)",
               "groups": {g: len(i) for g, i in GROUPS.items()}, "results": results, "runtime_s": round(time.time() - t0, 1)}
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "feature_ablation.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out / "feature_ablation.md").write_text(render(payload), encoding="utf-8")
    print(f"wrote {out / 'feature_ablation.md'}")
    return 0


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _d(x, base):
    if x is None or base is None or not (np.isfinite(x) and np.isfinite(base)):
        return "n/a"
    return f"{x - base:+.3f}"


def render(pl: dict) -> str:
    base = pl["results"]["none"]
    metrics = list(base)
    L = ["# Watchdog feature-group ablation (test split)", "",
         f"Watchdog `{pl['watchdog']}`. Each group is replaced by its training mean; the model is unchanged. "
         "Cells show the value after ablation and, in parentheses, the change from no ablation.", "",
         "| Ablated group | Features | " + " | ".join(metrics) + " |",
         "|---|---|" + "---|" * len(metrics)]
    L.append("| none | 0 | " + " | ".join(_f(base[m]) for m in metrics) + " |")
    for group, n in pl["groups"].items():
        r = pl["results"][group]
        L.append(f"| {group} | {n} | " + " | ".join(f"{_f(r[m])} ({_d(r[m], base[m])})" for m in metrics) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
