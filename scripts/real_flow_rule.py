"""Rule-based detector on real, unlabeled LOBSTER order flow, including SPY (execution plan, Phase 7; CP-4).

Usage:
    python data/download.py --spy --tickers      # SPY, 30 levels, 09:30-10:30 only (~940 MB)
    python scripts/real_flow_rule.py

The plan reserved SPY for a real clean-data false-positive test. Only the rule can run here: the Watchdog's
participant features need the identity of the trader behind each order, and LOBSTER messages carry none. Real data
has no manipulation labels, so the output is a flag RATE on real flow, not a false-positive rate.

Both rule settings used in this project are reported: N = 2 × depth with M = 200 events (tuned in
evaluation/basic_eval.py; the five-stock table in results/basic_eval.md) and M = 1000 events (tuned on the Watchdog
train split; every Watchdog-vs-rule comparison). Same 30,000-event warmup as the other stocks.

Writes results/real_flow_rule.md and results/real_flow_rule.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.cli import guard  # noqa: E402

from env.lobster_data import csv_paths, load_day  # noqa: E402
from env.normalization import reference_stats  # noqa: E402
from evaluation.baseline_detector import RuleDetector, real_data_flags  # noqa: E402

STOCKS = [("AAPL", 10), ("MSFT", 10), ("GOOG", 10), ("INTC", 10), ("AMZN", 10), ("SPY", 30)]
RULES = {"M=200 (basic_eval)": RuleDetector(2, 200), "M=1000 (train split)": RuleDetector(2, 1000)}


def main() -> int:
    guard(__doc__, protected=tuple(ROOT / "results" / f for f in ("real_flow_rule.json", "real_flow_rule.md")))
    rows = {}
    for tk, levels in STOCKS:
        if not all(p.exists() for p in csv_paths(tk, levels)):
            print(f"[skip] {tk}: data not downloaded")
            continue
        d = load_day(tk, levels)
        s = reference_stats(d)
        rows[tk] = {"levels": levels, "events": len(d), "window": "09:30-10:30" if tk == "SPY" else "09:30-16:00",
                    **{name: real_data_flags(d, s, det) for name, det in RULES.items()}}
        print(f"{tk}: " + "; ".join(f"{n} flags/10k={rows[tk][n]['flags_per_10k_events']:.2f}" for n in RULES), flush=True)
    out = ROOT / "results"
    payload = {"rules": {n: {"n_mult": r.n_mult, "m_events": r.m_events} for n, r in RULES.items()}, "stocks": rows}
    (out / "real_flow_rule.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out / "real_flow_rule.md").write_text(render(payload), encoding="utf-8")
    print(f"wrote {out / 'real_flow_rule.md'}")
    return 0


def render(pl: dict) -> str:
    L = ["# Rule on real, unlabeled order flow (including SPY)", "",
         "No labels exist, so these are flag **rates**, not false-positive rates. The Watchdog cannot run on real flow: "
         "LOBSTER has no participant identities, so its participant features cannot be built. Rule: size ≥ 2× trailing "
         "touch depth, deleted within M events without a visible execution first. Events after a 30,000-event warmup.", "",
         "| Stock | Window | Events after warmup | New orders | Large orders | Flags, M=200 | per 10k events | "
         "Flags, M=1000 | per 10k events | Share of large orders flagged (M=1000) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for tk, r in pl["stocks"].items():
        a, b = r["M=200 (basic_eval)"], r["M=1000 (train split)"]
        share = b["flags"] / b["large_orders"] if b["large_orders"] else float("nan")
        L.append(f"| {tk} | {r['window']} | {a['events']:,} | {a['new_orders']:,} | {a['large_orders']:,} | {a['flags']:,} | "
                 f"{a['flags_per_10k_events']:.2f} | {b['flags']:,} | {b['flags_per_10k_events']:.2f} | {100 * share:.0f}% |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
