"""Build the report figures from committed training logs and results.

Usage:
    python scripts/make_figures.py

Writes PNGs to results/figures/. Every figure is drawn from checkpoints/*/progress.csv or results/*.json,
so nothing is hand-drawn or illustrative. Figures that need a training log missing from a given machine are
skipped with a message rather than faked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.cli import guard  # noqa: E402
FIG = ROOT / "results" / "figures"
RESULTS = ROOT / "results"
WD, RULE = "#2a6fb5", "#d08b1e"
GOOD, BAD, GREY = "#2e8b57", "#c0392b", "#8a93a6"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9.5,
    "axes.grid": True, "grid.alpha": .25, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "figure.autolayout": True,
})


def load(name: str) -> dict:
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def progress(agent: str, tag: str) -> pd.DataFrame | None:
    p = ROOT / "checkpoints" / agent / tag / "progress.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df if len(df) else None


def smooth(y: pd.Series, k: int = 5) -> pd.Series:
    return y.rolling(k, min_periods=1).mean()


def fig_spoofer_training() -> str | None:
    """The credit-assignment fix: one decision per event never learns; one per 10 events does."""
    slow, fast = progress("SPOOFER-04", "v3_eps1"), progress("SPOOFER-04", "main")
    fast2 = progress("SPOOFER-02", "main")
    if slow is None or fast is None:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.5))
    for df, label, color, style in ((slow, "1 event per decision (failed)", BAD, "-"),
                                    (fast, "10 events per decision (final)", GOOD, "-")):
        d = df.dropna(subset=["ep_pnl_mean"])
        ax1.plot(d.timesteps / 1000, smooth(d.ep_pnl_mean) / 1000, style, color=color, lw=1.6, label=label)
    ax1.axhline(0, color=GREY, lw=.8, ls="--")
    ax1.set_xlabel("training steps (thousands)"); ax1.set_ylabel("profit per episode ($ thousands)")
    ax1.set_title("(a) SPOOFER-04 (INTC): the exploration fix")
    ax1.legend(loc="upper left")

    for df, label, color in ((fast, "SPOOFER-04 · INTC", WD), (fast2, "SPOOFER-02 · MSFT", RULE)):
        if df is None:
            continue
        d = df.dropna(subset=["ep_pnl_mean"])
        ax2.plot(d.timesteps / 1000, smooth(d["manip_trades_per_1k"]), color=color, lw=1.6, label=label)
    ax2.set_xlabel("training steps (thousands)")
    ax2.set_ylabel("trades against own resting order\n(per 1,000 steps)")
    ax2.set_title("(b) Spoofing behaviour emerging")
    ax2.legend(loc="lower right")
    out = FIG / "fig1_spoofer_training.png"
    fig.savefig(out); plt.close(fig)
    return out.name


def fig_watchdog_training() -> str | None:
    """Observation scaling decides whether the detector learns at all."""
    raw, norm = progress("WATCHDOG", "exp_raw_fp1"), progress("WATCHDOG", "main")
    if raw is None or norm is None:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.5))
    for df, label, color in ((raw, "raw observations (collapsed)", BAD), (norm, "standardised (final)", GOOD)):
        ax1.plot(df.timesteps / 1000, smooth(df.flag_rate, 7), color=color, lw=1.6, label=label)
    ax1.set_xlabel("training steps (thousands)"); ax1.set_ylabel("share of steps flagged")
    ax1.set_title("(a) Collapse to never flagging"); ax1.legend(loc="upper right")

    d = norm.dropna(subset=["precision"])
    ax2.plot(d.timesteps / 1000, smooth(d.precision, 7), color=WD, lw=1.6, label="precision")
    ax2.plot(d.timesteps / 1000, smooth(d.recall, 7), color=RULE, lw=1.6, label="recall")
    ax2.plot(d.timesteps / 1000, smooth(d.fpr, 7), color=BAD, lw=1.3, ls="--", label="false-positive rate")
    ax2.set_ylim(-.02, 1.02)
    ax2.set_xlabel("training steps (thousands)"); ax2.set_ylabel("rate")
    ax2.set_title("(b) Watchdog learning curve (final run)"); ax2.legend(loc="center right")
    out = FIG / "fig2_watchdog_training.png"
    fig.savefig(out); plt.close(fig)
    return out.name


def fig_detection() -> str | None:
    we, lb, lay, se, ms = (load(n) for n in ("watchdog_eval.json", "lateburst_test.json",
                                             "layering_test.json", "synthetic_eval.json", "multiseed.json"))
    if not (we and lb and lay):
        return None
    ind = we["buckets"]["In-distribution (test split)"]
    syn = se.get("detection", {}).get("Manipulators + legitimate", {})
    groups = [
        ("In-distribution\nF1", ind["watchdog_orders"]["f1"], ind["rule_orders"]["f1"]),
        ("Unseen attacker\nrecall", lb["overall"]["watchdog_orders"]["recall"], lb["overall"]["rule_orders"]["recall"]),
        ("Layering\nrecall", lay["pooled"]["watchdog_orders"]["recall"], lay["pooled"]["rule_orders"]["recall"]),
        ("Synthetic market\nF1", syn.get("watchdog_orders", {}).get("f1", np.nan), syn.get("rule_orders", {}).get("f1", np.nan)),
        ("False positives\non legitimate", we["buckets"]["Legitimate only (HONEST + FLICKER)"]["watchdog_orders"]["fpr"],
         we["buckets"]["Legitimate only (HONEST + FLICKER)"]["rule_orders"]["fpr"]),
    ]
    labels = [g[0] for g in groups]
    w = [g[1] for g in groups]; r = [g[2] for g in groups]
    x = np.arange(len(groups)); width = .36
    fig, ax = plt.subplots(figsize=(9.4, 3.8))
    b1 = ax.bar(x - width / 2, w, width, label="Watchdog (RL)", color=WD)
    b2 = ax.bar(x + width / 2, r, width, label="Rule baseline", color=RULE)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", fontsize=8.5, padding=2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.30); ax.set_ylabel("score")
    ax.set_title("Watchdog vs rule baseline — the surveillance trade-off")
    ax.legend(loc="upper left", ncol=2, fontsize=8.5)
    ax.annotate("higher is better", xy=(.28, .95), xycoords="axes fraction", fontsize=8, color=GREY)
    ax.annotate("lower is better", xy=(.845, .95), xycoords="axes fraction", fontsize=8, color=GREY, ha="center")
    out = FIG / "fig3_detection_comparison.png"
    fig.savefig(out); plt.close(fig)
    return out.name


def fig_impact() -> str | None:
    cal = ROOT / "configs" / "calibration.json"
    if not cal.exists():
        return None
    c = json.loads(cal.read_text(encoding="utf-8"))["tickers"]
    horizons = [10, 50, 200]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for tk, color in (("MSFT", WD), ("INTC", "#4b9ad4")):
        ramp = {int(k): v for k, v in c[tk]["ramp"].items()}
        ws = sorted(ramp)
        ys = [c[tk]["lambda"] * 10 * np.interp(h, [0] + ws, [0.0] + [ramp[w] for w in ws]) for h in horizons]
        ax.plot(horizons, ys, "o-", color=color, lw=1.8, label=f"replay model · {tk}")
    for tag, pf, color in (("pf005", "0.05", "#7fb069"), ("pf010", "0.10", GOOD), ("pf030", "0.30", "#b5651d")):
        j = load(f"synthetic_spoof_impact_{tag}.json")
        if not j:
            continue
        ys = [j["impact_by_horizon"][str(h)]["mean_ticks"] for h in horizons]
        es = [j["impact_by_horizon"][str(h)]["ci95_ticks"] for h in horizons]
        ax.errorbar(horizons, ys, yerr=es, fmt="s--", color=color, lw=1.4, capsize=3,
                    label=f"synthetic market · {pf} followers")
    ax.set_xlabel("market events after the spoof is placed"); ax.set_ylabel("price move (spreads)")
    ax.set_title("A spoof moves the price far less when the market\nreacts through order matching")
    ax.legend(fontsize=8)
    out = FIG / "fig4_spoof_impact.png"
    fig.savefig(out); plt.close(fig)
    return out.name


def fig_seeds() -> str | None:
    sd = load("spoofer_seeds.json")
    if not sd:
        return None
    rows = sd["per_seed"]
    labels = [f"{r['agent'].split('-')[1]}·{r['tag']}\n{r['ticker']}" for r in rows]
    vals = [r["calibrated"]["pnl_mean"] / 1000 for r in rows]
    errs = [r["calibrated"]["pnl_ci95"] / 1000 for r in rows]
    noimp = [r["no_impact"]["pnl_mean"] / 1000 for r in rows]
    x = np.arange(len(rows)); width = .36
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    b1 = ax.bar(x - width / 2, vals, width, yerr=errs, capsize=3, label="with price impact", color=WD)
    b2 = ax.bar(x + width / 2, noimp, width, label="impact switched off (counterfactual)", color=BAD)
    ax.bar_label(b1, fmt="%.1f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.1f", fontsize=8, padding=-12, color="white")
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylabel("profit per episode ($ thousands)")
    ax.set_title("Every seed's profit comes from the spoof")
    ax.legend(loc="lower left", fontsize=8.5)
    out = FIG / "fig5_seed_robustness.png"
    fig.savefig(out); plt.close(fig)
    return out.name


def fig_suppression() -> str | None:
    ps = load("pnl_suppression.json")
    if not ps:
        return None
    rows = [r for r in ps["rows"] if r["source"] in ("SPOOFER-04", "SPOOFER-02")]
    labels = [f"{r['source']}\n{r['ticker']}" for r in rows]
    none = [r["none"]["pnl_mean"] / 1000 for r in rows]
    wd = [r["watchdog"]["pnl_mean"] / 1000 for r in rows]
    rl = [r["rule"]["pnl_mean"] / 1000 for r in rows]
    x = np.arange(len(rows)); width = .26
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for off, vals, lab, col in ((-width, none, "no surveillance", GREY), (0, wd, "Watchdog acting", WD), (width, rl, "rule acting", RULE)):
        b = ax.bar(x + off, vals, width, label=lab, color=col)
        ax.bar_label(b, fmt="%.1f", fontsize=8, padding=2)
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("profit per episode ($ thousands)")
    ax.set_title("Surveillance acting inside the episode cuts the profit")
    ax.legend(fontsize=8.5)
    out = FIG / "fig6_pnl_suppression.png"
    fig.savefig(out); plt.close(fig)
    return out.name


def main() -> int:
    guard(__doc__)
    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in (fig_spoofer_training, fig_watchdog_training, fig_detection, fig_impact, fig_seeds, fig_suppression):
        try:
            name = fn()
        except Exception as exc:  # a missing column or log should not kill the rest
            print(f"[skip] {fn.__name__}: {exc}")
            continue
        if name:
            made.append(name)
            print(f"wrote results/figures/{name}")
        else:
            print(f"[skip] {fn.__name__}: required log or result missing")
    print(f"{len(made)} figures written to {FIG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
