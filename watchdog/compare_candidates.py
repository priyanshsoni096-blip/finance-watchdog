"""Compare candidate Watchdogs on the TEST split only (CP-7 development decisions).

Usage:
    python watchdog/compare_candidates.py --tags main main_s2 main_s3 causal causal_s2 causal_s3 [--name cp7_label]

Held-out and LATEBURST data are never loaded: a model change must be chosen without seeing them (HANDOFF §9), and
the chosen model is then scored once on a new pre-registered attacker. Per tag, reports order-level F1 / precision
/ recall in distribution, order recall per Spoofer (SPOOFER-03 GOOG is the sparse one), the order-level FPR on
legitimate test sources, and the flag rate on steps where an opposite trade happens against a resting
manipulative order (the act itself; see scripts/flag_timing_diagnostic.py). Groups of tags sharing a prefix up to
"_s<seed>" are summarised with mean ± 95% Student-t CI.

Writes results/<name>.md and results/<name>.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.baseline_detector import tune  # noqa: E402
from watchdog.dataset import TRADE_COL, load_split, orders_from_array  # noqa: E402
from watchdog.evaluate_watchdog import combine, score_source  # noqa: E402
from watchdog.multiseed_summary import mean_ci95  # noqa: E402
from watchdog.watchdog_env import load_obs_norm  # noqa: E402

SPOOFERS = ("SPOOFER-02_MSFT", "SPOOFER-03_GOOG", "SPOOFER-04_INTC")


def act_flag_rate(src: dict, step_flags: np.ndarray) -> tuple[int, int]:
    """(flags, steps) over steps on which the owner trades against a resting manipulative order."""
    off, hits, n = src["ep_offsets"], 0, 0
    for ep, o in orders_from_array(src["orders"]):
        if not o.manipulative:
            continue
        a = off[ep]
        window = np.arange(a + o.placed_step, a + o.removed_step)
        act = window[src["X"][window, TRADE_COL] == -o.side]
        hits += int(step_flags[act].sum())
        n += len(act)
    return hits, n


def evaluate(tag: str, test: dict, det) -> dict:
    from sb3_contrib import RecurrentPPO
    path = ROOT / "checkpoints" / "WATCHDOG" / tag / "model.zip"
    model = RecurrentPPO.load(path, device="cpu")
    obs_norm = load_obs_norm(path.parent)
    scored = {n: score_source(model, s, det, obs_norm) for n, s in test.items()}
    ind = combine(list(scored.values()))
    legit = combine([v for n, v in scored.items() if n.startswith(("HONEST", "FLICKER"))])
    row = {"order F1": ind["watchdog_orders"]["f1"], "order precision": ind["watchdog_orders"]["precision"],
           "order recall": ind["watchdog_orders"]["recall"], "legitimate order FPR": legit["watchdog_orders"]["fpr"]}
    for n in SPOOFERS:
        if n in scored:
            row[f"{n} order recall"] = combine([scored[n]])["watchdog_orders"]["recall"]
            h, k = act_flag_rate(test[n], scored[n]["step_flags"])
            row[f"{n} flag rate on the act"] = h / k if k else float("nan")
    return row


def group_of(tag: str) -> str:
    return re.sub(r"_s\d+$", "", tag)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--name", default="cp7_candidates")
    args = p.parse_args()

    train, test = load_split("train"), load_split("test")
    if not (train and test):
        print("missing train/test splits; run watchdog/dataset.py first")
        return 1
    det, _ = tune([o for s in train.values() for _, o in orders_from_array(s["orders"])])
    t0, per_tag = time.time(), {}
    for tag in args.tags:
        per_tag[tag] = evaluate(tag, test, det)
        print(f"[{time.time() - t0:5.0f}s] {tag}: " + ", ".join(f"{k}={v:.3f}" for k, v in per_tag[tag].items()), flush=True)
    groups: dict[str, list[str]] = {}
    for tag in per_tag:
        groups.setdefault(group_of(tag), []).append(tag)
    metrics = list(next(iter(per_tag.values())))
    summary = {g: {m: dict(zip(("mean", "ci95"), mean_ci95([per_tag[t][m] for t in tags]))) for m in metrics}
               for g, tags in groups.items()}
    payload = {"split": "test", "rule": {"n_mult": det.n_mult, "m_events": det.m_events}, "per_tag": per_tag,
               "groups": groups, "summary": summary, "runtime_s": round(time.time() - t0, 1)}
    out = ROOT / "results"
    (out / f"{args.name}.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out / f"{args.name}.md").write_text(render(payload), encoding="utf-8")
    print(f"wrote {out / (args.name + '.md')}")
    return 0


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def render(pl: dict) -> str:
    tags = list(pl["per_tag"])
    metrics = list(pl["per_tag"][tags[0]])
    L = ["# Candidate Watchdogs on the test split (CP-7 development)", "",
         "Test split only; held-out and pre-registered data were not loaded. Order level unless stated. "
         "\"Flag rate on the act\" is the share of steps, on which the owner trades against its own resting "
         "manipulative order, that the Watchdog flags.", "",
         "## Per model", "", "| Metric | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    for m in metrics:
        L.append(f"| {m} | " + " | ".join(_f(pl["per_tag"][t][m]) for t in tags) + " |")
    groups = list(pl["summary"])
    L += ["", "## Per group (mean ± 95% Student-t CI over seeds)", "",
          "| Metric | " + " | ".join(f"{g} (n={len(pl['groups'][g])})" for g in groups) + " |",
          "|---|" + "---|" * len(groups)]
    for m in metrics:
        L.append(f"| {m} | " + " | ".join(f"{_f(pl['summary'][g][m]['mean'])} ± {_f(pl['summary'][g][m]['ci95'])}"
                                         for g in groups) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
