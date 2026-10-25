"""Metrics in results/ must be internally consistent and agree across files.

Every reported precision, recall, F1 and false-positive rate is recomputed here from the confusion counts in
the same file, and values that appear in more than one file are cross-checked. This is the guard that stops a
future change from silently introducing conflicting numbers — the failure mode that produced the apparent
"0.22 vs 0.24" LATEBURST discrepancy, which turned out to be a 3-seed mean versus a single-seed run.
"""
import json
import math
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parent.parent / "results"
TOL = 5e-4


def load(name):
    p = RESULTS / name
    if not p.exists():
        pytest.skip(f"{name} not generated")
    return json.loads(p.read_text(encoding="utf-8"))


def iter_blocks(obj, path=""):
    """Yield (path, block) for every dict holding a confusion matrix."""
    if isinstance(obj, dict):
        if {"tp", "fp", "fn", "tn"} <= obj.keys():
            yield path, obj
        for k, v in obj.items():
            yield from iter_blocks(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_blocks(v, f"{path}[{i}]")


FILES = ["watchdog_eval.json", "lateburst_test.json", "layering_test.json", "synthetic_eval.json",
         "basic_eval.json"]


@pytest.mark.parametrize("fname", FILES)
def test_confusion_matrices_match_their_reported_rates(fname):
    data = load(fname)
    checked = 0
    for path, b in iter_blocks(data):
        tp, fp, fn, tn = b["tp"], b["fp"], b["fn"], b["tn"]
        if "n" in b:
            assert b["n"] == tp + fp + fn + tn, f"{fname}{path}: n != tp+fp+fn+tn"
        if "positives" in b:
            assert b["positives"] == tp + fn, f"{fname}{path}: positives != tp+fn"
        for key, num, den in (("precision", tp, tp + fp), ("recall", tp, tp + fn), ("fpr", fp, fp + tn)):
            if key not in b:
                continue
            got = b[key]
            if den == 0:
                assert got is None or (isinstance(got, float) and math.isnan(got)), \
                    f"{fname}{path}: {key} should be undefined with an empty denominator"
            else:
                assert abs(got - num / den) < TOL, f"{fname}{path}: {key} {got} != {num}/{den}"
                checked += 1
        if "f1" in b and b["f1"] == b["f1"] and tp:
            p, r = tp / (tp + fp), tp / (tp + fn)
            assert abs(b["f1"] - 2 * p * r / (p + r)) < TOL, f"{fname}{path}: f1 inconsistent"
            checked += 1
    assert checked > 0, f"{fname}: no metrics found to verify"


def test_multiseed_means_match_their_per_seed_values():
    ms = load("multiseed.json")
    for metric, summary in ms["summary"].items():
        vals = [ms["per_seed"][t][metric] for t in ms["tags"]]
        finite = [v for v in vals if v is not None and v == v]
        if not finite:
            continue
        assert abs(summary["mean"] - sum(finite) / len(finite)) < TOL, f"{metric}: mean != mean(per-seed)"


def test_lateburst_recall_agrees_across_files():
    """The single-seed file and the 'main' entry of the multi-seed file describe the same run."""
    lb, ms = load("lateburst_test.json"), load("multiseed.json")
    single = lb["overall"]["watchdog_orders"]["recall"]
    main_seed = ms["per_seed"]["main"]["LATEBURST-ATK order recall"]
    assert abs(single - main_seed) < TOL, "lateburst_test.json disagrees with multiseed 'main'"
    # and the 3-seed mean is a different, also-correct quantity — not equal unless the seeds agree
    assert ms["summary"]["LATEBURST-ATK order recall"]["mean"] == pytest.approx(
        sum(ms["per_seed"][t]["LATEBURST-ATK order recall"] for t in ms["tags"]) / len(ms["tags"]), abs=TOL)


def test_rule_baseline_is_identical_wherever_it_is_reported():
    """The rule is deterministic, so the same bucket must give the same number in every file."""
    we, ms = load("watchdog_eval.json"), load("multiseed.json")
    pairs = [("in-distribution order F1", we["buckets"]["In-distribution (test split)"]["rule_orders"]["f1"]),
             ("legitimate-only order FPR",
              we["buckets"]["Legitimate only (HONEST + FLICKER)"]["rule_orders"]["fpr"])]
    for key, from_eval in pairs:
        assert abs(ms["rule"][key] - from_eval) < TOL, f"rule {key} differs between files"


def test_suppression_percentages_match_the_pnl_they_summarise():
    ps = load("pnl_suppression.json")
    for row in ps["rows"]:
        base = row["none"]["pnl_mean"]
        for cond in ("watchdog", "rule"):
            got = row[cond].get("suppression_pct")
            if got is None or got != got:
                continue
            expected = 100.0 * (base - row[cond]["pnl_mean"]) / base
            assert abs(got - expected) < 0.05, f"{row['source']}/{cond}: suppression % inconsistent"


def test_demo_episode_summaries_match_their_recorded_steps():
    demo = load("demo_episodes.json")
    for ep in demo["episodes"]:
        s, steps, orders = ep["summary"], ep["steps"], ep["orders"]
        assert s["steps"] == len(steps)
        assert s["watchdog_flagged_steps"] == sum(x["watchdog_flag"] for x in steps)
        assert s["manipulation_steps"] == sum(x["manipulation_active"] for x in steps)
        assert s["large_orders"] == len(orders)
        assert s["manipulative_orders"] == sum(o["manipulative"] for o in orders)
        assert s["rule_false_alarms"] == sum(1 for o in orders if o["rule_flagged"] and not o["manipulative"])
