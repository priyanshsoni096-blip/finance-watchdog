"""Is reinforcement learning necessary? Supervised detectors on the same orders, same buckets.

The Watchdog never sees a label — it learns from a reward. A supervised classifier trained directly on the
hindsight labels is therefore not a competing research contribution but the obvious cheaper alternative, and
the comparison it enables is the one a reviewer will ask for: does the RL formulation buy anything that
logistic regression and gradient boosting on the same 46 features do not?

Protocol, identical to the Watchdog's wherever it can be:
  * features      the same surveillance observation, plus the mean of the previous 10 steps
  * fitting       the `train` split only
  * threshold     chosen on the `train` split only, maximising order-level F1; never retuned on test or heldout
  * decisions     a step flag becomes an order flag exactly as for the Watchdog (flagged at any resting step)
  * buckets       imported from watchdog/evaluate_watchdog.py, so rows line up with the published table

Usage:
    python scripts/supervised_baseline.py [--data data/watchdog] [--out results] [--tag main]

Writes results/supervised_baseline.json and .md. The Watchdog column is read from results/watchdog_eval.json
(same data, same buckets) rather than re-scored here.
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
sys.path.insert(0, str(ROOT / "scripts"))

from feature_signal_check import windowed  # noqa: E402
from watchdog.dataset import DATA_DIR, load_split, orders_from_array  # noqa: E402
from watchdog.evaluate_watchdog import BUCKETS, counts, order_flags  # noqa: E402

from evaluation.baseline_detector import RuleDetector, tune  # noqa: E402

GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def episode_slices(src: dict) -> list[tuple[int, int]]:
    off = src["ep_offsets"]
    return list(zip(off[:-1], off[1:]))


def order_decisions(prob: np.ndarray, src: dict, threshold: float) -> tuple[list[bool], list[bool]]:
    """-> (order flags, order truth), using the Watchdog's own step-flag -> order-flag rule."""
    flags_by_ep = [prob[a:b] > threshold for a, b in episode_slices(src)]
    orders = orders_from_array(src["orders"])
    dec, _ = order_flags(orders, flags_by_ep)
    return dec, [o.manipulative for _, o in orders]


def pick_threshold(probs: dict[str, np.ndarray], train: dict[str, dict]) -> tuple[float, float]:
    """Order-level F1 on the TRAIN split only. No test or heldout data is consulted."""
    best, best_f1 = 0.5, -1.0
    for th in GRID:
        pred, truth = [], []
        for name, src in train.items():
            d, t = order_decisions(probs[name], src, th)
            pred += d
            truth += t
        f1 = counts(pred, truth)["f1"]
        if np.isfinite(f1) and f1 > best_f1:
            best, best_f1 = float(th), float(f1)
    return best, best_f1


def fit_models(Xtr: np.ndarray, ytr: np.ndarray, seed: int = 0) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {}
    t0 = time.time()
    models["logistic regression"] = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
    ).fit(Xtr, ytr)
    print(f"[{time.time() - t0:5.0f}s] fitted logistic regression", flush=True)
    models["gradient boosting"] = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=seed,
        early_stopping=True, validation_fraction=0.1,
    ).fit(Xtr, ytr)
    print(f"[{time.time() - t0:5.0f}s] fitted gradient boosting", flush=True)
    return models


def score_all(model, groups: dict[tuple[str, str], dict]) -> dict[tuple[str, str], np.ndarray]:
    return {k: model.predict_proba(windowed(src))[:, 1] for k, src in groups.items()}


def evaluate(probs, groups, threshold, det: RuleDetector) -> dict:
    """Order-level counts per bucket for the supervised model, with the rule on the same orders."""
    scored = {}
    for key, src in groups.items():
        dec, truth = order_decisions(probs[key], src, threshold)
        orders = orders_from_array(src["orders"])
        scored[key] = {"pred": dec, "truth": truth, "rule": [det.flags(o) for _, o in orders],
                       "step_pred": (probs[key] > threshold), "step_true": src["y"].astype(bool)}
    out = {}
    for name, pred_fn in BUCKETS:
        members = [v for k, v in scored.items() if pred_fn(*k)]
        if not members:
            continue
        cat = lambda k: [x for m in members for x in m[k]]  # noqa: E731
        out[name] = {
            "orders": counts(cat("pred"), cat("truth")),
            "rule_orders": counts(cat("rule"), cat("truth")),
            "steps": counts(np.concatenate([m["step_pred"] for m in members]),
                            np.concatenate([m["step_true"] for m in members])),
        }
    return out


def _f(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def render(pl: dict) -> str:
    L = ["# Is RL necessary? Supervised detectors on the same orders", "",
         "The Watchdog learns from a reward and never sees a label. These classifiers are fitted directly on the "
         "hindsight labels, so they are an upper bound on what the 46 surveillance features support, not a "
         "competing contribution. Every model is scored on the same recorded orders, in the same buckets, with "
         "its decision threshold chosen on the train split alone.", ""]
    if pl.get("thresholds"):
        L += ["Thresholds (fitted on train orders): "
              + ", ".join(f"{k} = {v['threshold']:.2f} (train F1 {v['train_f1']:.2f})"
                          for k, v in pl["thresholds"].items()), ""]
    L += ["## Order-level F1", "",
          "| Bucket | Manipulative / orders | Rule | Logistic regression | Gradient boosting | RL Watchdog |",
          "|---|---|---|---|---|---|"]
    models = list(pl["models"])
    for bucket in pl["buckets"]:
        first = pl["models"][models[0]][bucket]
        cells = [_f(pl["models"][m][bucket]["orders"]["f1"]) for m in models]
        wd = pl["watchdog_f1"].get(bucket)
        L.append(f"| {bucket} | {first['orders']['positives']} / {first['orders']['n']} | "
                 f"{_f(first['rule_orders']['f1'])} | " + " | ".join(cells) + f" | {_f(wd)} |")
    for m in models:
        L += ["", f"## {m} — full order-level detail", "",
              "| Bucket | Precision | Recall | F1 | FPR |", "|---|---|---|---|---|"]
        for bucket, b in pl["models"][m].items():
            o = b["orders"]
            L.append(f"| {bucket} | {_f(o['precision'])} | {_f(o['recall'])} | {_f(o['f1'])} | {_f(o['fpr'], 3)} |")
    L += ["", "## Reading this table", "",
          "A supervised model that beats the RL Watchdog in-distribution shows the features carry the signal and "
          "the RL formulation is paying an optimisation cost for it. A supervised model that also collapses on "
          "the held-out attackers shows the generalisation failure is a property of the features and labels, not "
          "of reinforcement learning. The two conclusions are different and the table distinguishes them.", ""]
    return "\n".join(L) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--watchdog-eval", default=None, help="json to read the Watchdog column from")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data = Path(args.data)
    train, test, held = load_split("train", data), load_split("test", data), load_split("heldout", data)
    if not (train and test and held):
        print(f"missing splits under {data}; run watchdog/dataset.py first")
        return 1

    t0 = time.time()
    Xtr = np.vstack([windowed(s) for s in train.values()])
    ytr = np.concatenate([s["y"] for s in train.values()]).astype(bool)
    print(f"[{time.time() - t0:5.0f}s] train matrix {Xtr.shape}, positive rate {ytr.mean():.3f}", flush=True)

    models = fit_models(Xtr, ytr, args.seed)
    del Xtr
    det, _ = tune([o for s in train.values() for _, o in orders_from_array(s["orders"])])

    groups = {("test", n): s for n, s in test.items()} | {("heldout", n): s for n, s in held.items()}
    payload = {"data": str(data), "rule": {"n_mult": det.n_mult, "m_events": det.m_events},
               "train_sources": len(train), "train_steps": int(len(ytr)),
               "models": {}, "thresholds": {}, "watchdog_f1": {}}
    for name, model in models.items():
        train_probs = {n: model.predict_proba(windowed(s))[:, 1] for n, s in train.items()}
        th, train_f1 = pick_threshold(train_probs, train)
        payload["thresholds"][name] = {"threshold": th, "train_f1": train_f1}
        print(f"[{time.time() - t0:5.0f}s] {name}: threshold {th:.2f} (train order F1 {train_f1:.3f})", flush=True)
        payload["models"][name] = evaluate(score_all(model, groups), groups, th, det)

    wd_path = Path(args.watchdog_eval) if args.watchdog_eval else Path(args.out) / "watchdog_eval.json"
    if wd_path.exists():
        wd = json.loads(wd_path.read_text(encoding="utf-8"))
        payload["watchdog_eval"] = str(wd_path)
        payload["watchdog_f1"] = {k: v["watchdog_orders"]["f1"] for k, v in wd["buckets"].items()}
    payload["buckets"] = list(next(iter(payload["models"].values())))
    payload["runtime_s"] = round(time.time() - t0, 1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "supervised_baseline.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out / "supervised_baseline.md").write_text(render(payload), encoding="utf-8")
    print(f"wrote {out / 'supervised_baseline.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
