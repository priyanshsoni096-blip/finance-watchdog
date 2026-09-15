"""Evaluate the WATCHDOG against the rule-based baseline on identical recorded orders.

Usage:
    python watchdog/evaluate_watchdog.py [--tag main]

Buckets follow the evaluation matrix in the design:
  In-distribution          test split (same sources as training, unseen episodes)
  Held-out RL, AMZN        SPOOFER-05 + HONEST/FLICKER on AMZN  (unseen strategy AND stock)
  Scripted, train stocks   SCRIPTED-ATK on AAPL/MSFT/GOOG/INTC + FLICKER negatives (unseen structure)
  Scripted, AMZN           SCRIPTED-ATK + HONEST/FLICKER on AMZN
  Legitimate only          every HONEST and FLICKER source (false-positive check)

Order-level decision: the Watchdog "flags an order" if it flags at any step while the order rests;
the baseline flags at cancel time using the rule tuned on the TRAIN split only. Detection delay is
steps from placement to the first flag (baseline: the order's lifetime, since it can only fire on
the cancel). The Watchdog can also flag steps with no large order resting at all, so episodes of
HONEST traders with any flag are reported separately.

Writes results/watchdog_eval.json and results/watchdog_eval.md.
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

from evaluation.baseline_detector import OrderRecord, RuleDetector, tune  # noqa: E402
from watchdog.dataset import DATA_DIR, load_split, orders_from_array  # noqa: E402
from watchdog.watchdog_env import flag_episode  # noqa: E402

TRAIN_TICKERS = ["AAPL", "MSFT", "GOOG", "INTC"]


def order_flags(orders: list[tuple[int, OrderRecord]], flags_by_ep: list[np.ndarray]) -> tuple[list[bool], list[float]]:
    """Watchdog order-level decisions and detection delays (NaN when never flagged)."""
    decisions, delays = [], []
    for ep, o in orders:
        window = flags_by_ep[ep][o.placed_step:o.removed_step]
        hit = np.flatnonzero(window)
        decisions.append(bool(hit.size))
        delays.append(float(hit[0]) if hit.size else float("nan"))
    return decisions, delays


def counts(pred, truth) -> dict:
    pred, truth = np.asarray(pred, dtype=bool), np.asarray(truth, dtype=bool)
    tp, fp = int(np.sum(pred & truth)), int(np.sum(pred & ~truth))
    fn, tn = int(np.sum(~pred & truth)), int(np.sum(~pred & ~truth))
    nan = float("nan")
    precision = tp / (tp + fp) if tp + fp else nan
    recall = tp / (tp + fn) if tp + fn else nan
    f1 = 2 * precision * recall / (precision + recall) if tp else (0.0 if tp + fp + fn else nan)
    return {"n": len(pred), "positives": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else nan}


def score_source(model, src: dict, det: RuleDetector) -> dict:
    off = src["ep_offsets"]
    flags_by_ep = [flag_episode(model, src["X"][a:b]) for a, b in zip(off[:-1], off[1:])]
    orders = orders_from_array(src["orders"])
    wd_dec, wd_delay = order_flags(orders, flags_by_ep)
    truth = [o.manipulative for _, o in orders]
    return {
        "step_flags": np.concatenate(flags_by_ep) if flags_by_ep else np.zeros(0, bool),
        "step_labels": src["y"],
        "episodes_flagged": float(np.mean([f.any() for f in flags_by_ep])) if flags_by_ep else float("nan"),
        "order_truth": truth,
        "wd_orders": wd_dec,
        "wd_delay": wd_delay,
        "rule_orders": [det.flags(o) for _, o in orders],
        "rule_delay": [float(o.lifetime) if det.flags(o) else float("nan") for _, o in orders],
    }


def combine(scored: list[dict]) -> dict:
    cat = lambda k: [x for s in scored for x in s[k]]  # noqa: E731
    truth = cat("order_truth")
    wd_delay = np.array([d for d, t, f in zip(cat("wd_delay"), truth, cat("wd_orders")) if t and f])
    rule_delay = np.array([d for d, t, f in zip(cat("rule_delay"), truth, cat("rule_orders")) if t and f])
    steps_pred = np.concatenate([s["step_flags"] for s in scored])
    steps_true = np.concatenate([s["step_labels"] for s in scored])
    return {
        "watchdog_orders": counts(cat("wd_orders"), truth),
        "rule_orders": counts(cat("rule_orders"), truth),
        "watchdog_steps": counts(steps_pred, steps_true),
        "watchdog_median_delay": float(np.median(wd_delay)) if wd_delay.size else float("nan"),
        "rule_median_delay": float(np.median(rule_delay)) if rule_delay.size else float("nan"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="main")
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--out", default=str(ROOT / "results"))
    args = p.parse_args()

    from sb3_contrib import RecurrentPPO
    path = ROOT / "checkpoints" / "WATCHDOG" / args.tag / "model.zip"
    if not path.exists():
        print(f"no Watchdog at {path}; run watchdog/train_watchdog.py first")
        return 1
    model = RecurrentPPO.load(path, device="cpu")
    data = Path(args.data)
    train, test, held = load_split("train", data), load_split("test", data), load_split("heldout", data)
    if not (train and test and held):
        print("missing splits; run watchdog/dataset.py first")
        return 1

    det, _ = tune([o for s in train.values() for _, o in orders_from_array(s["orders"])])
    t0 = time.time()
    scored = {}
    for split, group in (("test", test), ("heldout", held)):
        for name, src in group.items():
            scored[(split, name)] = score_source(model, src, det)
            print(f"[{time.time() - t0:5.0f}s] scored {split}/{name}", flush=True)

    def pick(pred):
        return [v for k, v in scored.items() if pred(*k)]

    buckets = {
        "In-distribution (test split)": pick(lambda sp, n: sp == "test"),
        "Held-out RL: SPOOFER-05 + AMZN controls": pick(lambda sp, n: sp == "heldout" and n.endswith("_AMZN") and not n.startswith("SCRIPTED")),
        "Scripted attacker, train stocks + FLICKER": pick(lambda sp, n: (sp == "heldout" and n.startswith("SCRIPTED") and not n.endswith("_AMZN"))
                                                         or (sp == "test" and n.startswith("FLICKER"))),
        "Scripted attacker, AMZN + AMZN controls": pick(lambda sp, n: sp == "heldout" and n.endswith("_AMZN") and not n.startswith("SPOOFER")),
        "Legitimate only (HONEST + FLICKER)": pick(lambda sp, n: n.startswith(("HONEST", "FLICKER"))),
    }
    bucket_rows = {name: combine(v) for name, v in buckets.items() if v}
    sources = {f"{sp}/{n}": {"steps": int(len(s["step_labels"])), "positive_rate": float(np.mean(s["step_labels"])),
                             "watchdog_flag_rate": float(np.mean(s["step_flags"])),
                             "episodes_with_any_flag": s["episodes_flagged"],
                             "orders": len(s["order_truth"]), "manipulative_orders": int(np.sum(s["order_truth"]))}
               for (sp, n), s in scored.items()}

    payload = {"watchdog": str(path.relative_to(ROOT)), "rule": {"n_mult": det.n_mult, "m_events": det.m_events},
               "buckets": bucket_rows, "sources": sources, "runtime_s": round(time.time() - t0, 1)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "watchdog_eval.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out / "watchdog_eval.md").write_text(render(payload), encoding="utf-8")
    print(f"wrote {out / 'watchdog_eval.md'}")
    return 0


def _f(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def render(pl: dict) -> str:
    r = pl["rule"]
    L = ["# Watchdog vs rule-based baseline", "",
         "This project demonstrates the feasibility of an RL-based surveillance approach in a realistic "
         "limit-order-book simulation, and evaluates the structural similarity of the emergently-learned "
         "manipulation strategy to documented real-world manipulation cases.", "",
         f"Watchdog: `{pl['watchdog']}`. Baseline rule tuned on the train split: size ≥ {r['n_mult']}× depth, "
         f"cancelled within {r['m_events']} events. Both are scored on the same recorded orders. "
         "An order is manipulative if its owner traded on the opposite side while it rested.", "",
         "## Order-level detection", "",
         "| Bucket | Manipulative / orders | Watchdog P | Watchdog R | Watchdog F1 | Watchdog FPR | Rule P | Rule R | Rule F1 | Rule FPR | Median delay (steps) W / R |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, b in pl["buckets"].items():
        w, rl = b["watchdog_orders"], b["rule_orders"]
        L.append(f"| {name} | {w['positives']} / {w['n']} | {_f(w['precision'])} | {_f(w['recall'])} | {_f(w['f1'])} | "
                 f"{_f(w['fpr'])} | {_f(rl['precision'])} | {_f(rl['recall'])} | {_f(rl['f1'])} | {_f(rl['fpr'])} | "
                 f"{_f(b['watchdog_median_delay'], 0)} / {_f(b['rule_median_delay'], 0)} |")
    L += ["", "## Step-level Watchdog flags", "",
          "| Bucket | Positive steps / steps | Precision | Recall | FPR |", "|---|---|---|---|---|"]
    for name, b in pl["buckets"].items():
        s = b["watchdog_steps"]
        L.append(f"| {name} | {s['positives']:,} / {s['n']:,} | {_f(s['precision'])} | {_f(s['recall'])} | {_f(s['fpr'], 3)} |")
    L += ["", "## Per source", "",
          "| Source | Steps | Positive rate | Watchdog flag rate | Episodes with any flag | Orders | Manipulative |",
          "|---|---|---|---|---|---|---|"]
    for name, s in pl["sources"].items():
        L.append(f"| {name} | {s['steps']:,} | {s['positive_rate']:.3f} | {s['watchdog_flag_rate']:.3f} | "
                 f"{_f(s['episodes_with_any_flag'])} | {s['orders']} | {s['manipulative_orders']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
