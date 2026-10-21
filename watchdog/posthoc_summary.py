"""Summarise the post-hoc synthetic-market, layering and PnL-suppression runs across Watchdog seeds (CP-7 follow-up,
CP-5).

Usage (after the runs that write results/posthoc/<tag>/):
    python synthetic/evaluate_synthetic.py --tag cov --out results/posthoc/cov        (and _s2, _s3)
    python synthetic/layering_test.py --tag cov --out results/posthoc/cov             (and _s2, _s3)
    python watchdog/pnl_suppression.py --tag cov --out results/posthoc/cov            (and main_s2, main_s3, cov_s2, cov_s3)
    python watchdog/posthoc_summary.py

The `main` seed reads the committed results/*.json. LAYER-ATK and the synthetic attackers have been inspected, so every
number for a model other than `main` is post-hoc. PnL suppression across three seeds is CP-5 of the execution plan.

Writes results/posthoc_summary.md and results/posthoc_summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from watchdog.multiseed_summary import mean_ci95  # noqa: E402

GROUPS = {"main": ["main", "main_s2", "main_s3"], "cov": ["cov", "cov_s2", "cov_s3"]}


def _load(tag: str, name: str) -> dict | None:
    f = (ROOT / "results" / f"{name}.json") if tag == "main" else (ROOT / "results" / "posthoc" / tag / f"{name}.json")
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def per_seed(tag: str) -> dict:
    out = {}
    syn = _load(tag, "synthetic_eval")
    if syn:
        d = syn["detection"]
        for k in ("precision", "recall", "f1"):
            out[f"synthetic: all manipulators {k}"] = d["Manipulators + legitimate"]["watchdog_orders"][k]
        out["synthetic: legitimate FPR"] = d["Legitimate only"]["watchdog_orders"]["fpr"]
        for src in ("SPOOFER-02", "SPOOFER-04", "SCRIPTED-ATK", "LATEBURST-ATK"):
            out[f"synthetic: {src} F1"] = d[f"{src} + legitimate"]["watchdog_orders"]["f1"]
    lay = _load(tag, "layering_test")
    if lay:
        out["LAYER-ATK recall"] = lay["pooled"]["watchdog_orders"]["recall"]
        out["LAYER-ATK pooled F1"] = lay["pooled"]["watchdog_orders"]["f1"]
        out["LAYER-ATK FLICKER FPR"] = lay["flicker_only"]["watchdog_orders"]["fpr"]
    pnl = _load(tag, "pnl_suppression")
    if pnl:
        for r in pnl["rows"]:
            key = f"{r['source']} {r['ticker']}"
            if r["source"].startswith("SPOOFER"):
                out[f"PnL suppression: {key} (%)"] = r["watchdog"]["suppression_pct"]
                out[f"PnL with Watchdog: {key} ($)"] = r["watchdog"]["pnl_mean"]
            if r["source"] == "FLICKER":
                out[f"interventions/episode on legitimate: {key}"] = r["watchdog"]["triggers_per_ep"]
    return out


def rule_row() -> dict:
    out = {}
    syn, lay, pnl = _load("main", "synthetic_eval"), _load("main", "layering_test"), _load("main", "pnl_suppression")
    d = syn["detection"]
    for k in ("precision", "recall", "f1"):
        out[f"synthetic: all manipulators {k}"] = d["Manipulators + legitimate"]["rule_orders"][k]
    out["synthetic: legitimate FPR"] = d["Legitimate only"]["rule_orders"]["fpr"]
    for src in ("SPOOFER-02", "SPOOFER-04", "SCRIPTED-ATK", "LATEBURST-ATK"):
        out[f"synthetic: {src} F1"] = d[f"{src} + legitimate"]["rule_orders"]["f1"]
    out["LAYER-ATK recall"] = lay["pooled"]["rule_orders"]["recall"]
    out["LAYER-ATK pooled F1"] = lay["pooled"]["rule_orders"]["f1"]
    out["LAYER-ATK FLICKER FPR"] = lay["flicker_only"]["rule_orders"]["fpr"]
    for r in pnl["rows"]:
        key = f"{r['source']} {r['ticker']}"
        if r["source"].startswith("SPOOFER"):
            out[f"PnL suppression: {key} (%)"] = r["rule"]["suppression_pct"]
            out[f"PnL with Watchdog: {key} ($)"] = r["rule"]["pnl_mean"]
        if r["source"] == "FLICKER":
            out[f"interventions/episode on legitimate: {key}"] = r["rule"]["triggers_per_ep"]
    return out


def main() -> int:
    seeds = {t: per_seed(t) for ts in GROUPS.values() for t in ts}
    rule = rule_row()
    metrics = list(rule)
    summary = {g: {m: dict(zip(("mean", "ci95"), mean_ci95([seeds[t].get(m) for t in ts if m in seeds[t]])))
                   for m in metrics} for g, ts in GROUPS.items()}
    counts = {g: {m: sum(m in seeds[t] for t in ts) for m in metrics} for g, ts in GROUPS.items()}
    no_surv = {f"{r['source']} {r['ticker']}": (r["none"]["pnl_mean"], r["none"]["pnl_ci95"])
               for r in _load("main", "pnl_suppression")["rows"] if r["source"].startswith("SPOOFER")}
    payload = {"post_hoc": True, "groups": GROUPS, "per_seed": seeds, "summary": summary, "seed_counts": counts,
               "rule": rule, "pnl_no_surveillance": no_surv}
    out = ROOT / "results"
    (out / "posthoc_summary.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out / "posthoc_summary.md").write_text(render(payload), encoding="utf-8")
    print(f"wrote {out / 'posthoc_summary.md'}")
    return 0


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{nd}f}"


def render(pl: dict) -> str:
    L = ["# Synthetic market, layering and PnL suppression across Watchdog seeds", "",
         "**Post-hoc** for every model other than `main` (all attackers here were inspected earlier). "
         "`main` = original Watchdog, `cov` = CP-7 round 2. Mean ± 95% Student-t CI over seeds; n is the number of "
         "seeds with that measurement (synthetic and layering were run for `main` once, so it has no CI there). "
         "In PnL rows the rule column is the rule acting as surveillance.", "",
         "PnL without surveillance (same seeds, 50 episodes): " +
         ", ".join(f"{k} {m:+,.0f} ± {h:,.0f}" for k, (m, h) in pl["pnl_no_surveillance"].items()) + ".", "",
         "| Metric | main | cov | Rule |", "|---|---|---|---|"]
    for m, rv in pl["rule"].items():
        cells = []
        for g in ("main", "cov"):
            s, n = pl["summary"][g][m], pl["seed_counts"][g][m]
            cells.append(f"{_f(s['mean'])} ± {_f(s['ci95'])} (n={n})" if n > 1 else f"{_f(s['mean'])} (n={n})")
        L.append(f"| {m} | {cells[0]} | {cells[1]} | {_f(rv)} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
