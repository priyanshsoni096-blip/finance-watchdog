"""One consolidated table of every evaluation bucket, Watchdog vs rule (execution plan, Phases 3/4/7).

Usage:
    python scripts/consolidate_results.py

Reads the committed results/*.json (nothing is re-run) and writes, for the main Watchdog and the train-tuned rule:
  * one row per bucket with the full confusion matrix (TP/FP/FN/TN), precision, recall, F1, FPR and median
    detection delay, covering replay in-distribution, held-out RL, scripted attacker, legitimate-only,
    LATEBURST-ATK, synthetic market and LAYER-ATK
  * the multi-seed summary, PnL suppression and the rule's flag rate on real unlabeled order flow

Sensitivity check requested by the plan's risk register ("tune N/M on held-out data too, so the comparison isn't
rigged either direction"): the rule is re-tuned on each replay bucket's OWN orders and scored on them. This is an
oracle upper bound for the rule, not a valid detector, and it never replaces the train-tuned rule, which is the
comparison every pre-registered verdict used. It needs data/watchdog/{train,test,heldout,lateburst}
(watchdog/dataset.py and watchdog/lateburst_test.py); the synthetic buckets are skipped because their orders are
not saved.

Writes results/consolidated.md and results/consolidated.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.cli import guard  # noqa: E402

from evaluation.baseline_detector import classification_metrics, tune  # noqa: E402
from watchdog.dataset import load_split, orders_from_array  # noqa: E402
from watchdog.evaluate_watchdog import BUCKETS  # noqa: E402

RESULTS = ROOT / "results"
STEP_EVENTS = 10  # every source decides once per 10 market events, so delays in steps x 10 = market events


def _load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def detection_rows() -> list[dict]:
    """(market, bucket, source file, bucket dict with watchdog_orders / rule_orders / delays)."""
    wd, late, syn, lay = _load("watchdog_eval"), _load("lateburst_test"), _load("synthetic_eval"), _load("layering_test")
    rows = [("replay", name, "watchdog_eval", b) for name, b in wd["buckets"].items()]
    rows.append(("replay", "LATEBURST-ATK (pre-registered) + FLICKER, all stocks", "lateburst_test", late["overall"]))
    rows += [("synthetic", f"{name}", "synthetic_eval", b) for name, b in syn["detection"].items()
             if name in ("Manipulators + legitimate", "Legitimate only")]
    rows.append(("synthetic", "LAYER-ATK (pre-registered) + FLICKER", "layering_test", lay["pooled"]))
    return [{"market": m, "bucket": n, "source": s, "watchdog": b["watchdog_orders"], "rule": b["rule_orders"],
             "watchdog_median_delay_steps": b["watchdog_median_delay"], "rule_median_delay_steps": b["rule_median_delay"]}
            for m, n, s, b in rows]


def oracle_rule_rows() -> list[dict]:
    """Rule re-tuned on each replay bucket's own orders (upper bound for the rule)."""
    test, held, late = load_split("test"), load_split("heldout"), load_split("lateburst")
    if not (test and held and late):
        return []
    tagged = [("test", n, s) for n, s in test.items()] + [("heldout", n, s) for n, s in held.items()]
    buckets = {name: [o for sp, n, s in tagged if pred(sp, n) for _, o in orders_from_array(s["orders"])]
               for name, pred in BUCKETS}
    buckets["LATEBURST-ATK (pre-registered) + FLICKER, all stocks"] = [
        o for s in late.values() for _, o in orders_from_array(s["orders"])]
    out = []
    for name, orders in buckets.items():
        det, _ = tune(orders)
        m = classification_metrics(orders, det)
        out.append({"bucket": name, "n_mult": det.n_mult, "m_events": det.m_events,
                    **{k: m[k] for k in ("tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr")}})
    return out


def main() -> int:
    guard(__doc__)
    ms, pnl, basic = _load("multiseed"), _load("pnl_suppression"), _load("basic_eval")
    payload = {
        "rule": _load("watchdog_eval")["rule"],
        "detection": detection_rows(),
        "oracle_rule": oracle_rule_rows(),
        "multiseed": {"tags": ms["tags"], "summary": ms["summary"], "rule": ms["rule"]},
        "pnl_suppression": [{k: r[k] for k in ("source", "ticker", "kind", "none", "watchdog", "rule")}
                            for r in pnl["rows"]],
        "real_flow": basic["real_flow"],
    }
    (RESULTS / "consolidated.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (RESULTS / "consolidated.md").write_text(render(payload), encoding="utf-8")
    if not payload["oracle_rule"]:
        print("oracle-rule rows skipped: need data/watchdog/{test,heldout,lateburst}")
    print(f"wrote {RESULTS / 'consolidated.md'}")
    return 0


def _f(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _pct(x) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.0f}%"


def _cm(m: dict) -> str:
    return f"{m['tp']} / {m['fp']} / {m['fn']} / {m['tn']}"


def render(pl: dict) -> str:
    r = pl["rule"]
    L = ["# Consolidated results: Watchdog vs rule-based baseline", "",
         "This project demonstrates the feasibility of an RL-based surveillance approach in a realistic "
         "limit-order-book simulation, and evaluates the structural similarity of the emergently-learned "
         "manipulation strategy to documented real-world manipulation cases.", "",
         "Built by `scripts/consolidate_results.py` from the committed `results/*.json`; no numbers are re-estimated. "
         f"Main Watchdog (`checkpoints/WATCHDOG/main`) vs the rule tuned on the replay train split "
         f"(size ≥ {r['n_mult']}× depth, cancelled within {r['m_events']} events). Order level. "
         "An order is manipulative if its owner traded on the opposite side while it rested. "
         f"Delays are agent steps of {STEP_EVENTS} market events from placement to first flag.", "",
         "## Detection, every bucket", "",
         "Confusion matrix is TP / FP / FN / TN.", "",
         "| Market | Bucket | Watchdog TP/FP/FN/TN | Watchdog P | R | F1 | FPR | Rule TP/FP/FN/TN | Rule P | R | F1 | FPR | Median delay W / R | Source |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in pl["detection"]:
        w, rl = row["watchdog"], row["rule"]
        L.append(f"| {row['market']} | {row['bucket']} | {_cm(w)} | {_f(w['precision'])} | {_f(w['recall'])} | "
                 f"{_f(w['f1'])} | {_f(w['fpr'])} | {_cm(rl)} | {_f(rl['precision'])} | {_f(rl['recall'])} | "
                 f"{_f(rl['f1'])} | {_f(rl['fpr'])} | {_f(row['watchdog_median_delay_steps'], 0)} / "
                 f"{_f(row['rule_median_delay_steps'], 0)} | `{row['source']}.json` |")
    L += ["", "## Three Watchdog seeds (mean ± 95% Student-t CI)", "",
          f"Seeds: {', '.join(pl['multiseed']['tags'])}. Source: `multiseed.json`.", "",
          "| Metric | Watchdog | Rule |", "|---|---|---|"]
    for m, s in pl["multiseed"]["summary"].items():
        L.append(f"| {m} | {_f(s['mean'], 3)} ± {_f(s['ci95'], 3)} | {_f(pl['multiseed']['rule'].get(m), 3)} |")
    L += ["", "## PnL suppression (main Watchdog, surveillance acting inside episodes)", "",
          "Mean PnL per episode ± 95% CI; suppression only meaningful where PnL without surveillance is clearly "
          "positive. Source: `pnl_suppression.json`.", "",
          "| Source | Stock | Kind | No surveillance | Watchdog | Suppression | Rule | Suppression |",
          "|---|---|---|---|---|---|---|---|"]
    for row in pl["pnl_suppression"]:
        n, w, rl = row["none"], row["watchdog"], row["rule"]
        L.append(f"| {row['source']} | {row['ticker']} | {row['kind']} | {n['pnl_mean']:+,.0f} ± {n['pnl_ci95']:,.0f} | "
                 f"{w['pnl_mean']:+,.0f} ± {w['pnl_ci95']:,.0f} | {_pct(w.get('suppression_pct'))} | "
                 f"{rl['pnl_mean']:+,.0f} ± {rl['pnl_ci95']:,.0f} | {_pct(rl.get('suppression_pct'))} |")
    L += ["", "## Rule on real, unlabeled LOBSTER order flow", "",
          "No labels exist, so this is a flag rate, not a false-positive rate. The Watchdog cannot run here: "
          "LOBSTER has no participant identities. Source: `basic_eval.json`.", "",
          "| Stock | Events | Large orders | Flags | Flags per 10k events |", "|---|---|---|---|---|"]
    for tk, x in pl["real_flow"].items():
        L.append(f"| {tk} | {x['events']:,} | {x['large_orders']:,} | {x['flags']:,} | {x['flags_per_10k_events']:.2f} |")
    if pl["oracle_rule"]:
        L += ["", "## Sensitivity: rule re-tuned on each bucket's own orders (oracle upper bound)", "",
              "Requested by the execution plan's risk register. Each bucket's rule parameters are chosen with "
              "hindsight on the very orders it is scored on, so this is the best any (N, M) rule could do, "
              "not a deployable detector. It does not change any pre-registered verdict. Buckets with no manipulative "
              "orders have no F1, so their parameters are just the grid's most conservative choice.", "",
              "| Bucket | Oracle N × depth | Oracle M events | TP/FP/FN/TN | P | R | F1 | FPR |",
              "|---|---|---|---|---|---|---|---|"]
        for row in pl["oracle_rule"]:
            L.append(f"| {row['bucket']} | {row['n_mult']} | {row['m_events']} | {_cm(row)} | {_f(row['precision'])} | "
                     f"{_f(row['recall'])} | {_f(row['f1'])} | {_f(row['fpr'])} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
