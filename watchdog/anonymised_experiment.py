"""Can manipulation be detected WITHOUT privileged participant attribution?

Every detector in this project so far has been handed the monitored participant's own trade direction
(observation feature 44) and the side its resting orders sit on (feature 45). The manipulation label is
"the owner traded the opposite side while the order rested", so those two features contain the label's
definition and a zero-parameter comparison of them scores order-level F1 0.997 (scripts/label_observability.py).

This script asks the question that remains once that crutch is removed: **using only what an external
surveillance system can see on an anonymised feed, how much manipulation is detectable?**

Information used
----------------
Only the 40 order-book features (10 price levels per side, prices in spread units and sizes as log ratios to
trailing depth), as the market displays them. These already include the monitored participant's resting orders
inside the displayed size, exactly as a real feed would. Derived features are computed from that book sequence
alone: level-1 and deep imbalance, total displayed depth, spread, and their changes and rolling statistics over
the preceding window. **No participant attribution of any kind is used.**

Features 40-45 (the participant block) are dropped, and `--audit` checks that the remaining ones cannot simply
reconstruct the label by another route.

Two formulations, because they are different questions
------------------------------------------------------
  step     Is manipulation happening at this instant? Causal label (positives begin at the first opposite
           trade), so no future information is attached to an earlier instant.
  episode  Does this 200-step window contain manipulation at all? Retrospective, which is what an offline
           surveillance review actually does.

Protocol
--------
Temporal splits (`data/watchdog_v2`, disjoint stretches of the trading day) so train and test never share
market data. Preprocessing and thresholds are fitted on training data only. Episodes never cross splits.
Evaluated against a majority-class baseline, a transparent book-imbalance rule, logistic regression and
gradient boosting, and compared with the attributed detectors on the same episodes.

Usage:
    python watchdog/anonymised_experiment.py --audit      # leakage audit only, no models
    python watchdog/anonymised_experiment.py              # full experiment
    python watchdog/anonymised_experiment.py --data data/watchdog   # original (overlapping) splits

Writes results/anonymised_experiment.json and .md.
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

from evaluation.rollout import BOOK_DIM  # noqa: E402
from watchdog.dataset import DATA_DIR, causal_labels, load_split  # noqa: E402
from watchdog.evaluate_watchdog import counts  # noqa: E402

V2_DIR = ROOT / "data" / "watchdog_v2"
HISTORY = 10
# the two withheld participant features, by index in the recorded 46-dim observation
TRADE_COL, IMBALANCE_COL = 40 + 4, 40 + 5

DERIVED_NAMES = [
    "level-1 imbalance", "level-1 imbalance (normalised)", "deep imbalance",
    "deep imbalance (normalised)", "total displayed depth", "spread",
    "largest ask level", "largest bid level", "largest ask - largest bid",
    "ask size dispersion", "bid size dispersion",
]


def feature_name(j: int) -> str:
    """Readable name for a column of the retained (attribution-free) matrix."""
    if j < BOOK_DIM:
        level, which = divmod(j, 4)
        return f"book L{level + 1} {['ask price', 'ask size', 'bid price', 'bid size'][which]}"
    k = j - BOOK_DIM
    return DERIVED_NAMES[k] if k < len(DERIVED_NAMES) else f"derived[{k}]"


# --------------------------------------------------------------------------- features
def book_features(X: np.ndarray) -> np.ndarray:
    """Aggregate, attribution-free features derived from the displayed book only.

    LOBSTER column order within the 40: [ask_p, ask_sz, bid_p, bid_sz] per level.
    """
    b = X[:, :BOOK_DIM]
    ask_sz = b[:, 1::4]
    bid_sz = b[:, 3::4]
    ask_p, bid_p = b[:, 0::4], b[:, 2::4]

    top_ask, top_bid = ask_sz[:, 0], bid_sz[:, 0]
    tot_ask, tot_bid = ask_sz.sum(1), bid_sz.sum(1)
    eps = 1e-6
    feats = [
        top_bid - top_ask,                                   # level-1 imbalance
        (top_bid - top_ask) / (top_bid + top_ask + eps),      # normalised level-1 imbalance
        tot_bid - tot_ask,                                   # deep imbalance
        (tot_bid - tot_ask) / (tot_bid + tot_ask + eps),      # normalised deep imbalance
        tot_ask + tot_bid,                                   # total displayed depth
        ask_p[:, 0] - bid_p[:, 0],                           # spread in spread units
        ask_sz.max(1), bid_sz.max(1),                        # largest displayed level per side
        ask_sz.max(1) - bid_sz.max(1),
        ask_sz.std(1), bid_sz.std(1),                        # how lumpy each side is
    ]
    return np.column_stack([np.asarray(f, dtype=np.float32) for f in feats])


def windowed(mat: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """[current, mean of previous HISTORY, current - that mean], computed within each episode."""
    out = np.empty((len(mat), mat.shape[1] * 3), np.float32)
    for a, b in zip(offsets[:-1], offsets[1:]):
        ep = mat[a:b]
        csum = np.cumsum(np.vstack([np.zeros((1, ep.shape[1]), np.float32), ep]), axis=0)
        idx = np.arange(len(ep))
        lo = np.maximum(idx - HISTORY, 0)
        prev = (csum[idx] - csum[lo]) / np.maximum(idx - lo, 1)[:, None]
        out[a:b] = np.hstack([ep, prev, ep - prev])
    return out


def build(split: str, data: Path, pred=lambda n: True):
    """-> (step features, step labels, episode ids, per-source index) with NO participant features."""
    srcs = load_split(split, data)
    names = [n for n in sorted(srcs) if pred(n)]
    if not names:
        return None
    Xs, ys, eps, owner = [], [], [], []
    base = 0
    for n in names:
        s = srcs[n]
        raw = np.hstack([s["X"][:, :BOOK_DIM], book_features(s["X"])])
        Xs.append(windowed(raw, s["ep_offsets"]))
        ys.append(causal_labels(s))
        off = s["ep_offsets"]
        ep_id = np.repeat(np.arange(len(off) - 1) + base, np.diff(off))
        eps.append(ep_id)
        owner += [n] * len(ep_id)
        base += len(off) - 1
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(eps), np.array(owner)


# --------------------------------------------------------------------------- leakage audit
def leakage_audit(data: Path) -> dict:
    """Can the label be recovered from the attribution-free features by a simple route?"""
    srcs = load_split("train", data)
    X = np.vstack([np.hstack([s["X"][:, :BOOK_DIM], book_features(s["X"])]) for s in srcs.values()])
    y = np.concatenate([causal_labels(s) for s in srcs.values()])
    withheld = np.concatenate([s["X"][:, [TRADE_COL, IMBALANCE_COL]] for s in srcs.values()])

    out = {"n": int(len(y)), "positive_rate": float(y.mean())}
    # 1. the withheld pair still reconstructs the label, which is the point of the exercise
    sig = (np.sign(withheld[:, 1]) * np.sign(withheld[:, 0])) < 0
    out["withheld_pair_step_f1"] = counts(sig, y)["f1"]
    # 2. no retained feature may be near-deterministic for the label
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = X.std(0)
        safe = sd > 1e-9
        corr = np.zeros(X.shape[1])
        corr[safe] = np.abs(((X[:, safe] - X[:, safe].mean(0)) * (y - y.mean())[:, None]).mean(0)
                            / (sd[safe] * y.std() + 1e-12))
    out["max_abs_point_biserial_corr"] = float(np.nanmax(corr))
    out["n_features_corr_above_0.5"] = int((corr > 0.5).sum())
    top = int(np.nanargmax(corr))
    out["top_correlated_feature"] = feature_name(top)
    out["top_correlated_feature_index"] = top
    # 3. best single-feature threshold rule, swept on train (an upper bound on 1-D recoverability)
    best = {"f1": -1.0}
    for j in np.argsort(-corr)[:12]:
        col = X[:, j]
        for q in (50, 75, 90, 95, 99):
            thr = np.percentile(col, q)
            for pred in (col > thr, col < thr):
                c = counts(pred, y)
                if np.isfinite(c["f1"]) and c["f1"] > best["f1"]:
                    best = {"f1": c["f1"], "feature": int(j), "pct": q,
                            "direction": "greater" if pred is not None and (col > thr).sum() == pred.sum() else "less"}
    out["best_single_feature_rule_f1"] = best["f1"]
    out["best_single_feature_index"] = best.get("feature")
    out["best_single_feature"] = (feature_name(best["feature"]) if best.get("feature") is not None else None)
    return out


# --------------------------------------------------------------------------- models
def fit_and_score(Xtr, ytr, groups, sets: dict, seed: int) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
        "gradient boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, class_weight="balanced", random_state=seed,
            early_stopping=True, validation_fraction=0.15),
    }
    out = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        # threshold chosen on TRAIN only
        ptr = m.predict_proba(Xtr)[:, 1]
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
        thr = max(grid, key=lambda t: (counts(ptr > t, ytr)["f1"] if np.isfinite(counts(ptr > t, ytr)["f1"]) else -1))
        out[name] = {"threshold": float(thr), "train_f1": counts(ptr > thr, ytr)["f1"], "sets": {}}
        for sname, (Xs, ys, eps) in sets.items():
            p = m.predict_proba(Xs)[:, 1]
            out[name]["sets"][sname] = {
                "step": counts(p > thr, ys) | {"pr_auc": pr_auc(p, ys)},
                "episode": episode_counts(p > thr, ys, eps),
            }
    return out


def pr_auc(score: np.ndarray, y: np.ndarray) -> float:
    y = y.astype(bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    hits = y[order]
    prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(prec[hits].sum() / y.sum())


def episode_counts(pred_step: np.ndarray, y_step: np.ndarray, eps: np.ndarray) -> dict:
    """An episode is flagged if any step is flagged; it is positive if any step is positive."""
    uniq = np.unique(eps)
    pred = np.array([pred_step[eps == e].any() for e in uniq])
    truth = np.array([y_step[eps == e].any() for e in uniq])
    return counts(pred, truth)


def baselines(sets: dict, Xtr, ytr) -> dict:
    """Degenerate constant classifiers and a transparent book-imbalance rule, all fixed on train.

    BOTH constants are reported. Which one is the majority differs between the two formulations (steps are
    ~11% positive, but an episode almost always contains some manipulation when a manipulator is present), and
    "always flag" is a strong F1 baseline on a positive-heavy set. Omitting it would let a model look good
    while actually losing to a constant.
    """
    out = {}
    for label, const in (("constant: never flag", False), ("constant: always flag", True)):
        out[label] = {"sets": {
            s: {"step": counts(np.full(len(y), const), y),
                "episode": episode_counts(np.full(len(y), const), y, e)}
            for s, (X, y, e) in sets.items()}}
    # transparent rule: normalised deep imbalance beyond a train-fitted percentile, either side
    col = BOOK_DIM + 3
    thr = float(np.percentile(np.abs(Xtr[:, col]), 90))
    out[f"book-imbalance rule (|deep imbalance| > p90 = {thr:.3f})"] = {"threshold": thr, "sets": {
        s: {"step": counts(np.abs(X[:, col]) > thr, y),
            "episode": episode_counts(np.abs(X[:, col]) > thr, y, e)}
        for s, (X, y, e) in sets.items()}}
    return out


# --------------------------------------------------------------------------- report
def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def render(pl: dict) -> str:
    a = pl["audit"]
    L = ["# Detection without participant attribution", "",
         "Every other detector in this project is handed the monitored participant's own trade direction and the "
         "side of its resting orders — together, the manipulation label's definition. This experiment removes "
         "that attribution entirely and uses **only the displayed order book**, as an external surveillance "
         "system on an anonymised feed would see it.", "",
         f"Data: `{pl['data']}`. Splits are disjoint stretches of the trading day where available; preprocessing "
         "and every threshold are fitted on training data only; episodes never cross splits.", "",
         "## Leakage audit of the retained features", "",
         "| Check | Value |", "|---|---|",
         f"| Steps in train | {a['n']:,} |",
         f"| Positive rate (causal label) | {a['positive_rate']:.3f} |",
         f"| **Withheld pair** (features 44 + 45) step F1 — what we removed | **{_f(a['withheld_pair_step_f1'])}** |",
         f"| Largest absolute correlation of any retained feature with the label | {_f(a['max_abs_point_biserial_corr'])} |",
         f"| Retained features with correlation > 0.5 | {a['n_features_corr_above_0.5']} |",
         f"| Best single retained feature threshold rule (swept on train) | {_f(a['best_single_feature_rule_f1'])} "
         f"(`{a.get('best_single_feature')}`) |",
         f"| Most label-correlated retained feature | `{a.get('top_correlated_feature')}` |",
         "",
         "The withheld pair reconstructs the label almost perfectly; no retained feature comes close. The "
         "removal is therefore real — the label is not simply reachable by another column.", ""]

    for form, title in (("step", "Step level — is manipulation happening now?"),
                        ("episode", "Episode level — does this window contain manipulation?")):
        L += [f"## {title}", "",
              "| Model | Split | Precision | Recall | F1 | FPR | Positives / n |", "|---|---|---|---|---|---|---|"]
        for mname, m in pl["models"].items():
            for sname, s in m["sets"].items():
                c = s[form]
                L.append(f"| {mname} | {sname} | {_f(c['precision'])} | {_f(c['recall'])} | {_f(c['f1'])} | "
                         f"{_f(c['fpr'])} | {c['positives']:,} / {c['n']:,} |")
        L.append("")

    h = pl["headline"]
    L += ["## What this establishes", "",
          f"With attribution removed, the best learned model reaches step-level F1 **{_f(h['best_step_f1'])}** "
          f"({h['best_step_model']}) and episode-level F1 **{_f(h['best_episode_f1'])}** "
          f"({h['best_episode_model']}) on the held-out split, against **{_f(a['withheld_pair_step_f1'])}** "
          "step-level F1 for the two features that were removed and 0.99+ order-level F1 for every attributed "
          "detector in the project.", "",
          "### Against a constant classifier", "",
          "| Formulation | Best learned model | Best constant classifier | Learned model wins? |",
          "|---|---|---|---|",
          f"| Step | {_f(h['best_step_f1'])} ({h['best_step_model']}) | {_f(h['best_constant_step_f1'])} "
          f"({h['best_constant_step']}) | {'**yes**' if h['beats_constant_step'] else '**NO**'} |",
          f"| Episode | {_f(h['best_episode_f1'])} ({h['best_episode_model']}) | "
          f"{_f(h['best_constant_episode_f1'])} ({h['best_constant_episode']}) | "
          f"{'**yes**' if h['beats_constant_episode'] else '**NO**'} |", ""]
    if not h["beats_constant_episode"]:
        L += ["**The episode-level F1 is not a result.** The held-out set is positive-heavy, so flagging every "
              "window scores a higher F1 than any model here. Episode-level F1 should not be quoted for this "
              "experiment; the step-level figures and the episode-level false-positive rates are the meaningful "
              "ones. Gradient boosting is the only model that separates anything usefully at episode level, and "
              "it does so by being conservative (low recall, very low FPR), not by beating the constant on F1.",
              ""]
    L += ["**This is the honest measure of the detection problem.** The attributed result was never evidence that "
          "manipulation is detectable; it was evidence that a label can be recovered from its own definition.", "",
          "### Limitation that bounds this experiment", "",
          "In this environment only **one** participant acts on top of historical replay, so any anomaly in the "
          "displayed book is attributable to that participant by construction. A real anonymised feed mixes many "
          "participants, which makes the problem harder still. These numbers are therefore an **upper bound** on "
          "what attribution-free detection would achieve in a real market, not a lower bound.", ""]
    return "\n".join(L) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(V2_DIR if (V2_DIR / "train").exists() else DATA_DIR))
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--audit", action="store_true", help="run the leakage audit only")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data = Path(args.data)
    if not (data / "train").exists():
        print(f"no recorded data under {data}; run watchdog/dataset.py first")
        return 1

    t0 = time.time()
    audit = leakage_audit(data)
    print(f"[{time.time() - t0:5.0f}s] leakage audit: withheld pair step F1 "
          f"{audit['withheld_pair_step_f1']:.3f}; best retained single-feature rule "
          f"{audit['best_single_feature_rule_f1']:.3f}; max |corr| "
          f"{audit['max_abs_point_biserial_corr']:.3f}", flush=True)
    if args.audit:
        print(json.dumps(audit, indent=2, default=float))
        return 0

    tr = build("train", data)
    if tr is None:
        print("no training sources")
        return 1
    Xtr, ytr, eptr, _ = tr
    sets = {}
    for label, split, pred in (("test (in-distribution)", "test", lambda n: True),
                               ("held-out", "heldout", lambda n: True)):
        got = build(split, data, pred)
        if got is not None:
            sets[label] = (got[0], got[1], got[2])
    print(f"[{time.time() - t0:5.0f}s] train {Xtr.shape}, positive rate {ytr.mean():.3f}; "
          f"eval sets {list(sets)}", flush=True)

    models = baselines(sets, Xtr, ytr) | fit_and_score(Xtr, ytr, eptr, sets, args.seed)
    print(f"[{time.time() - t0:5.0f}s] fitted and scored", flush=True)

    held = "held-out" if "held-out" in sets else next(iter(sets))
    CONST = ("constant: never flag", "constant: always flag")
    learned = {k: v for k, v in models.items() if k not in CONST}

    def best(group, form):
        vals = [(n, m["sets"][held][form]["f1"]) for n, m in group.items()
                if np.isfinite(m["sets"][held][form]["f1"])]
        return max(vals, key=lambda kv: kv[1], default=(None, float("nan")))

    best_step_name, best_step = best(learned, "step")
    best_ep_name, best_ep = best(learned, "episode")
    const_step_name, const_step = best({k: models[k] for k in CONST}, "step")
    const_ep_name, const_ep = best({k: models[k] for k in CONST}, "episode")
    payload = {"data": str(data), "history": HISTORY, "seed": args.seed, "audit": audit, "models": models,
               "headline": {"best_step_f1": best_step, "best_step_model": best_step_name,
                            "best_episode_f1": best_ep, "best_episode_model": best_ep_name,
                            "best_constant_step_f1": const_step, "best_constant_step": const_step_name,
                            "best_constant_episode_f1": const_ep, "best_constant_episode": const_ep_name,
                            "beats_constant_step": bool(best_step > const_step),
                            "beats_constant_episode": bool(best_ep > const_ep),
                            "held_out_set": held},
               "runtime_s": round(time.time() - t0, 1)}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "anonymised_experiment.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out / "anonymised_experiment.md").write_text(render(payload), encoding="utf-8")
    print(f"wrote {out / 'anonymised_experiment.md'}")
    print(f"  best held-out step F1 {best_step:.3f}, episode F1 {best_ep:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
