"""POST-HOC re-scoring of several Watchdogs on every saved replay evaluation dataset (CP-7 follow-up).

Usage:
    python watchdog/posthoc_rescore.py [--tags main main_s2 main_s3 causal_g0_1m ... cov cov_s2 cov_s3]

Scores each tag on data already on disk: the test and heldout splits (buckets as in watchdog/evaluate_watchdog.py),
and the saved data of the three pre-registered replay tests (LATEBURST-ATK, DRIP-ATK, REVERSAL-ATK). Every one of
those attackers has now been inspected, so for any model other than the one each test was registered with, these
numbers are post-hoc: they describe the models and decide nothing. Tags sharing a prefix up to "_s<seed>" are
summarised as mean ± 95% Student-t CI; the rule is reported once.

Writes results/posthoc_rescore.md and results/posthoc_rescore.json.
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
from watchdog.dataset import load_split, orders_from_array  # noqa: E402
from watchdog.evaluate_watchdog import BUCKETS, combine, score_source  # noqa: E402
from watchdog.multiseed_summary import mean_ci95  # noqa: E402
from watchdog.watchdog_env import load_obs_norm  # noqa: E402

DEFAULT_TAGS = ["main", "main_s2", "main_s3", "causal_g0_1m", "causal_g0_1m_s2", "causal_g0_1m_s3",
                "cov", "cov_s2", "cov_s3"]
# pre-registered test -> (saved data folder, attacker source prefix, model it was registered with)
TESTS = {"LATEBURST-ATK": ("lateburst", "LATEBURST", "main"),
         "DRIP-ATK": ("drip", "DRIP", "causal_g0_1m"),
         "REVERSAL-ATK": ("reversal", "REVERSAL", "cov")}


def metrics(scored: dict) -> tuple[dict, dict]:
    """Watchdog and rule headline metrics from {(split, name): scored source}."""
    w, r = {}, {}
    rows = {name: combine([v for k, v in scored.items() if pred(*k)]) for name, pred in BUCKETS}
    ind = rows["In-distribution (test split)"]
    w["in-distribution order F1"], r["in-distribution order F1"] = ind["watchdog_orders"]["f1"], ind["rule_orders"]["f1"]
    for name, label in (("Legitimate only (HONEST + FLICKER)", "legitimate order FPR"),
                        ("Held-out RL: SPOOFER-05 + AMZN controls", "held-out AMZN legitimate FPR")):
        w[label], r[label] = rows[name]["watchdog_orders"]["fpr"], rows[name]["rule_orders"]["fpr"]
    for name, label in (("Scripted attacker, train stocks + FLICKER", "SCRIPTED-ATK train stocks"),
                        ("Scripted attacker, AMZN + AMZN controls", "SCRIPTED-ATK AMZN")):
        w[f"{label} recall"], r[f"{label} recall"] = rows[name]["watchdog_orders"]["recall"], rows[name]["rule_orders"]["recall"]
    for test, (folder, prefix, _) in TESTS.items():
        group = [v for (sp, n), v in scored.items() if sp == folder]
        if not group:
            continue
        both = combine(group)
        attack = combine([v for (sp, n), v in scored.items() if sp == folder and n.startswith(prefix)])
        legit = combine([v for (sp, n), v in scored.items() if sp == folder and not n.startswith(prefix)])
        for who, out in (("watchdog", w), ("rule", r)):
            out[f"{test} recall"] = attack[f"{who}_orders"]["recall"]
            out[f"{test} legitimate FPR"] = legit[f"{who}_orders"]["fpr"]
            out[f"{test} pooled F1"] = both[f"{who}_orders"]["f1"]
    return w, r


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    args = p.parse_args()
    from sb3_contrib import RecurrentPPO

    train = load_split("train")
    data = {sp: load_split(sp) for sp in ["test", "heldout"] + [f for f, _, _ in TESTS.values()]}
    missing = [sp for sp, d in data.items() if not d]
    if not train or missing:
        print(f"missing data: train={bool(train)} {missing}")
        return 1
    det, _ = tune([o for s in train.values() for _, o in orders_from_array(s["orders"])])
    t0, per_tag, rule = time.time(), {}, None
    for tag in args.tags:
        path = ROOT / "checkpoints" / "WATCHDOG" / tag / "model.zip"
        model, obs_norm = RecurrentPPO.load(path, device="cpu"), load_obs_norm(path.parent)
        scored = {(sp, n): score_source(model, src, det, obs_norm) for sp, group in data.items() for n, src in group.items()}
        per_tag[tag], rule = metrics(scored)
        print(f"[{time.time() - t0:5.0f}s] {tag}", flush=True)
    groups: dict[str, list[str]] = {}
    for tag in per_tag:
        groups.setdefault(re.sub(r"_s\d+$", "", tag), []).append(tag)
    names = list(next(iter(per_tag.values())))
    summary = {g: {m: dict(zip(("mean", "ci95"), mean_ci95([per_tag[t][m] for t in ts]))) for m in names}
               for g, ts in groups.items()}
    payload = {"post_hoc": True, "registered_with": {t: v[2] for t, v in TESTS.items()}, "per_tag": per_tag,
               "groups": groups, "summary": summary, "rule": rule,
               "rule_params": {"n_mult": det.n_mult, "m_events": det.m_events}, "runtime_s": round(time.time() - t0, 1)}
    out = ROOT / "results"
    (out / "posthoc_rescore.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out / "posthoc_rescore.md").write_text(render(payload), encoding="utf-8")
    print(f"wrote {out / 'posthoc_rescore.md'}")
    return 0


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def render(pl: dict) -> str:
    groups = list(pl["summary"])
    L = ["# Post-hoc re-scoring of Watchdog generations (replay market)", "",
         "**Post-hoc.** Every attacker below has been inspected. Each pre-registered verdict stands as the single "
         "run on the model it was registered with (" + ", ".join(f"{t}: `{m}`" for t, m in pl["registered_with"].items())
         + "). Any other model on that data is a description, not a test.", "",
         "Generations: `main` = original Watchdog; `causal_g0_1m` = CP-7 round 1 (causal label, γ 0); "
         "`cov` = CP-7 round 2 (plus varied-timing training sources). Mean ± 95% Student-t CI over 3 seeds. "
         f"Rule: size ≥ {pl['rule_params']['n_mult']}× depth, cancelled within {pl['rule_params']['m_events']} events.", "",
         "| Metric | " + " | ".join(f"{g} (n={len(pl['groups'][g])})" for g in groups) + " | Rule |",
         "|---|" + "---|" * len(groups) + "---|"]
    for m in pl["rule"]:
        cells = " | ".join(f"{_f(pl['summary'][g][m]['mean'])} ± {_f(pl['summary'][g][m]['ci95'])}" for g in groups)
        L.append(f"| {m} | {cells} | {_f(pl['rule'][m])} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
