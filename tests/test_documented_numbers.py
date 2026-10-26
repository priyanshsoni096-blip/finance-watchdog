"""Headline numbers quoted in the documentation must match the results files they come from.

Documentation drifts. A number gets improved by a re-run, the results file is committed, and a sentence three
documents away still quotes the old one. This project has had that happen more than once: the README quoted
the original Watchdog's 0.753 beside CP-7 KPIs, and the changelog claimed the layering test had not been
re-measured when it had.

Each case below pins one claim that appears in prose to the file that produced it. A re-run that moves a
number fails here, which is the point: the document has to be updated deliberately, not forgotten.
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load(name):
    p = RESULTS / name
    if not p.exists():
        pytest.skip(f"{name} not generated")
    return json.loads(p.read_text(encoding="utf-8"))


def doc(name: str) -> str:
    p = ROOT / name
    if not p.exists():
        pytest.skip(f"{name} missing")
    return p.read_text(encoding="utf-8")


def quoted(text: str, pattern: str) -> set[str]:
    """Every numeric string matching `pattern` in the document."""
    return set(re.findall(pattern, text))


# --------------------------------------------------------------------------- the label ceiling
def test_zero_parameter_signature_f1_is_quoted_correctly():
    """0.997 appears in the README, the audit, the report and the console. One source."""
    lo = load("label_observability.json")
    f1 = lo["signatures"]["opposite-side signature"]["in-distribution (test split)"]["orders"]["f1"]
    assert f"{f1:.3f}" == "0.997", f"source value moved to {f1:.4f}; every doc quoting 0.997 must change"
    for name in ("README.md", "docs/FINAL_AUDIT.md", "docs/MODEL_COMPARISON.md"):
        assert "0.997" in doc(name), f"{name} no longer quotes the ceiling F1"


def test_held_out_ceiling_is_quoted_correctly():
    lo = load("label_observability.json")
    f1 = lo["signatures"]["opposite-side signature"]["held-out scripted attacker"]["orders"]["f1"]
    assert f"{f1:.3f}" == "0.995", f"held-out ceiling moved to {f1:.4f}"


# --------------------------------------------------------------------------- CP-7 detector
def test_cp7_in_distribution_f1_matches_the_posthoc_summary():
    ph = load("posthoc_rescore.json")
    mu = ph["summary"]["cov"]["in-distribution order F1"]["mean"]
    assert f"{mu:.3f}" == "0.984", f"CP-7 in-distribution F1 moved to {mu:.4f}"
    assert "0.984" in doc("README.md"), "README no longer quotes the CP-7 in-distribution F1"


def test_original_watchdog_f1_is_still_quoted_as_history_not_as_current():
    ph = load("posthoc_rescore.json")
    mu = ph["summary"]["main"]["in-distribution order F1"]["mean"]
    assert f"{mu:.3f}" == "0.753"
    readme = doc("README.md")
    assert "0.753" in readme, "the original figure should remain, as history"
    # it must never be the lone headline: the CP-7 value has to appear too
    assert "0.984" in readme


@pytest.mark.parametrize("metric,expected", [
    ("LATEBURST-ATK recall", "0.984"),
    ("DRIP-ATK recall", "0.989"),
    ("REVERSAL-ATK recall", "0.983"),
])
def test_cp7_held_out_recalls_match_their_source(metric, expected):
    ph = load("posthoc_rescore.json")
    mu = ph["summary"]["cov"][metric]["mean"]
    assert f"{mu:.3f}" == expected, f"{metric} moved to {mu:.4f}; docs quoting {expected} must change"


# --------------------------------------------------------------------------- layering and suppression
def test_layering_recall_quoted_in_the_docs_matches_the_summary():
    ps = load("posthoc_summary.json")
    mu = ps["summary"]["cov"]["LAYER-ATK recall"]["mean"]
    assert f"{mu:.3f}" == "0.990", f"CP-7 layering recall moved to {mu:.4f}"
    assert "0.990" in doc("README.md"), "README no longer quotes the layering result"


def test_suppression_percentage_matches_the_summary():
    ps = load("posthoc_summary.json")
    mu = ps["summary"]["cov"]["PnL suppression: SPOOFER-04 INTC (%)"]["mean"]
    assert 100.0 < mu < 103.0, f"CP-7 suppression moved to {mu:.2f}%"
    assert "101" in doc("README.md")


def test_no_document_still_claims_layering_was_not_re_measured():
    for name in ("README.md", "REPORT.md", "HANDOFF.md", "docs/FINAL_AUDIT.md",
                 "docs/MODEL_COMPARISON.md"):
        assert "not re-measured" not in doc(name), f"{name} still says a CP-7 figure was not re-measured"


# --------------------------------------------------------------------------- attribution-free
def test_attribution_free_headline_matches_its_source():
    an = load("anonymised_experiment.json")
    h = an["headline"]
    assert f"{h['best_step_f1']:.3f}" == "0.392", f"attribution-free step F1 moved to {h['best_step_f1']:.4f}"
    assert f"{h['best_constant_step_f1']:.3f}" == "0.218"
    assert h["beats_constant_step"] is True, "the step-level result no longer beats a constant"
    assert h["beats_constant_episode"] is False, (
        "the episode-level result now beats a constant; the 'not a result' wording must be removed")


def test_withheld_pair_strength_matches():
    an = load("anonymised_experiment.json")
    assert f"{an['audit']['withheld_pair_step_f1']:.3f}" == "0.948"


# --------------------------------------------------------------------------- attacker
def test_spoofer_profit_quoted_in_the_readme_matches_the_seed_summary():
    sd = load("spoofer_seeds.json")
    intc = sd["across_seeds"]["SPOOFER-04"]
    assert 14000 < intc["pnl_mean"] < 14800, f"INTC profit moved to {intc['pnl_mean']:.0f}"
    assert intc["no_impact_mean"] < 0, "the impact-off counterfactual must still be a loss"
    assert "14,368" in doc("README.md"), "README no longer quotes the INTC profit"


def test_test_count_in_the_docs_matches_the_suite(request):
    """The number of collected tests must equal what the documents advertise.

    Only meaningful when the whole suite was collected; running one file would otherwise "fail" the docs for
    advertising more tests than this invocation happens to contain.
    """
    collected_files = {Path(str(i.fspath)).name for i in request.session.items}
    all_files = {p.name for p in (ROOT / "tests").glob("test_*.py")}
    if collected_files != all_files:
        pytest.skip(f"partial run ({len(collected_files)} of {len(all_files)} test files)")
    collected = len(request.session.items)
    for name in ("README.md", "docs/FINAL_AUDIT.md"):
        text = doc(name)
        numbers = {int(n) for n in re.findall(r"\b(\d{3})\s+tests\b", text)}
        if not numbers:
            continue
        assert collected in numbers, (
            f"{name} advertises {sorted(numbers)} tests but the suite collects {collected}")
