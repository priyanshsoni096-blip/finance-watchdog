"""How hard is the detection task, really? A two-feature, zero-parameter signature says: not hard.

The manipulation label is `opposite_trades > 0` — the order's owner traded on the opposite side while the
order rested. The surveillance observation hands the detector two features that describe exactly that event:

    feature 44  trade direction of the monitored participant this step (+1 buy, -1 sell, 0 none)
    feature 45  signed size imbalance of the participant's resting orders (which side it is resting on)

So "resting on one side while trading the other" is not a learned pattern; it is the label's own definition
expressed in the observation. This script measures that directly, with no fitting of any kind, and is the
honest upper bound every learned detector in this project must be read against.

The result is a limitation of the problem formulation, not a bug: the environment grants surveillance perfect
attribution of orders and trades to one monitored participant. Real surveillance must establish that
attribution first, which is the part this setup assumes away.

Usage:
    python scripts/label_observability.py [--data data/watchdog] [--out results]

Writes results/label_observability.json and .md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from watchdog.dataset import DATA_DIR, load_split, orders_from_array  # noqa: E402
from watchdog.evaluate_watchdog import counts, order_flags  # noqa: E402

TRADE_DIR, RESTING_IMBALANCE = 44, 45

SIGNATURES = {
    "opposite-side signature": lambda X: (np.sign(X[:, RESTING_IMBALANCE]) * np.sign(X[:, TRADE_DIR])) < 0,
    "any trade while resting": lambda X: (np.abs(X[:, TRADE_DIR]) > 0) & (np.abs(X[:, RESTING_IMBALANCE]) > 0),
    "resting order present": lambda X: np.abs(X[:, RESTING_IMBALANCE]) > 0,
}


def groups(data: Path) -> dict[str, dict]:
    test, held = load_split("test", data), load_split("heldout", data)
    train = load_split("train", data)
    return {
        "train (fitted by nothing)": train,
        "in-distribution (test split)": test,
        "held-out scripted attacker": {n: s for n, s in held.items() if n.startswith("SCRIPTED")},
        "held-out RL attacker (AMZN)": {n: s for n, s in held.items()
                                        if n.endswith("_AMZN") and not n.startswith("SCRIPTED")},
        "legitimate only (HONEST + FLICKER)": {n: s for n, s in (test | held).items()
                                               if n.startswith(("HONEST", "FLICKER"))},
    }


def score(flags_fn, group: dict) -> dict:
    """Step-level and order-level counts, using the Watchdog's own step-flag -> order-flag rule."""
    pred, truth = [], []
    step_pred, step_true = [], []
    for src in group.values():
        X, off = src["X"], src["ep_offsets"]
        f = flags_fn(X)
        orders = orders_from_array(src["orders"])
        d, _ = order_flags(orders, [f[a:b] for a, b in zip(off[:-1], off[1:])])
        pred += d
        truth += [o.manipulative for _, o in orders]
        step_pred.append(f)
        step_true.append(src["y"].astype(bool))
    return {"orders": counts(pred, truth),
            "steps": counts(np.concatenate(step_pred), np.concatenate(step_true))}


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def render(pl: dict) -> str:
    L = ["# The manipulation label is directly observable", "",
         "The label is `opposite_trades > 0`: the order's owner traded on the opposite side while the order "
         "rested. Observation feature 44 is the participant's trade direction this step and feature 45 is the "
         "side its resting orders sit on, so the label's definition is present in the observation. The "
         "signature below is a hand-written comparison of those two features. **Nothing is fitted — there are "
         "no parameters and no training.**", "",
         "## Order-level detection by a zero-parameter signature", "",
         "| Group | Manipulative / orders | Precision | Recall | F1 | FPR |", "|---|---|---|---|---|---|"]
    primary = pl["signatures"]["opposite-side signature"]
    for g, b in primary.items():
        o = b["orders"]
        L.append(f"| {g} | {o['positives']} / {o['n']} | {_f(o['precision'])} | {_f(o['recall'])} | "
                 f"{_f(o['f1'])} | {_f(o['fpr'])} |")
    L += ["", "## Step-level", "",
          "| Group | Positive steps / steps | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    for g, b in primary.items():
        s = b["steps"]
        L.append(f"| {g} | {s['positives']:,} / {s['n']:,} | {_f(s['precision'])} | {_f(s['recall'])} | "
                 f"{_f(s['f1'])} |")
    L += ["", "Step-level recall is below 1 because the label is assigned with hindsight to **every** step an "
          "order rests, while the signature fires only on the steps where the opposite trade actually occurs. "
          "Aggregating to the order ('flagged at any resting step') recovers it.", "",
          "## Other signatures, for contrast", "",
          "| Signature | In-distribution order F1 | Held-out scripted order F1 | Legitimate-only FPR |",
          "|---|---|---|---|"]
    for name, by_group in pl["signatures"].items():
        ind = by_group.get("in-distribution (test split)", {}).get("orders", {})
        sc = by_group.get("held-out scripted attacker", {}).get("orders", {})
        lg = by_group.get("legitimate only (HONEST + FLICKER)", {}).get("orders", {})
        L.append(f"| {name} | {_f(ind.get('f1'))} | {_f(sc.get('f1'))} | {_f(lg.get('fpr'))} |")
    L += ["", "## What this means", "",
          "1. **The detection task as posed is close to a tautology.** Given the observation, the label can be "
          "read off two features, so a learned detector is not discovering manipulation — at best it is "
          "recovering a definition it was handed.",
          "2. **Comparisons against the learned detector must be read in that light.** The rule baseline and the "
          "RL Watchdog both score far below this signature, so their gap measures the cost of their own "
          "restrictions (the rule may only fire at cancel time; the Watchdog is trained from a sparse reward "
          "rather than the label), not the difficulty of the problem.",
          "3. **The simplification is perfect attribution.** The environment tells surveillance which orders and "
          "trades belong to the monitored participant. Real surveillance must establish that first, and on "
          "anonymised data such as LOBSTER it cannot. That — not pattern recognition — is the hard part of "
          "spoofing detection, and this project assumes it away.",
          "4. **The generalisation failure is therefore about the learner, not the features.** A zero-parameter "
          "signature transfers to the unseen scripted attacker unchanged, so the Watchdog's collapse on held-out "
          "attackers cannot be explained by the observation lacking the signal.", ""]
    return "\n".join(L) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--out", default=str(ROOT / "results"))
    args = p.parse_args()

    data = Path(args.data)
    gs = {k: v for k, v in groups(data).items() if v}
    if not gs:
        print(f"no recorded data under {data}; run watchdog/dataset.py first")
        return 1

    payload = {"data": str(data), "features": {"trade_direction": TRADE_DIR, "resting_imbalance": RESTING_IMBALANCE},
               "signatures": {}}
    for name, fn in SIGNATURES.items():
        payload["signatures"][name] = {g: score(fn, grp) for g, grp in gs.items()}
        o = payload["signatures"][name]["in-distribution (test split)"]["orders"]
        print(f"{name:26s} in-distribution order F1 {o['f1']:.3f} "
              f"(P {o['precision']:.3f} R {o['recall']:.3f})", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "label_observability.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out / "label_observability.md").write_text(render(payload), encoding="utf-8")
    print(f"wrote {out / 'label_observability.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
