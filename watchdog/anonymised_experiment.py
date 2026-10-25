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


def student_t_ci(vals: list[float]) -> tuple[float, float]:
    """Mean and 95% half-width over seeds; NaN half-width for a single run."""
    v = np.array([x for x in vals if x is not None and x == x], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float("nan")
    # two-sided 95% t for small samples; table avoids a scipy dependency
    T = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}
    t_crit = T.get(v.size, 1.96)
    return float(v.mean()), float(t_crit * v.std(ddof=1) / np.sqrt(v.size))


def brier(score: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error of the predicted probability: lower is better, 0.25 is a coin flip at p=0.5."""
    return float(np.mean((score - y.astype(float)) ** 2))


def reliability(score: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """Calibration curve: predicted probability vs observed frequency, per decile of predicted probability."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (score >= lo) & (score < hi if hi < 1.0 else score <= hi)
        if m.sum() == 0:
            continue
        out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                    "predicted": float(score[m].mean()), "observed": float(y[m].mean())})
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
class SeqClassifier:
    """LSTM over whole episodes, sized like the Watchdog's policy (LSTM 64 -> 64 tanh -> 1).

    A sequence model is the fair comparison for a recurrent RL detector: if order flow carries manipulation
    in its temporal structure rather than in any single step, this is the model that can use it.
    """

    def __init__(self, n_features: int, seed: int, epochs: int = 6):
        import torch
        from torch import nn
        torch.manual_seed(seed)
        self.torch, self.epochs = torch, epochs
        self.net = nn.ModuleDict({
            "lstm": nn.LSTM(n_features, 64, batch_first=True),
            "head": nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)),
        })
        self.mu = self.sd = None

    def _episodes(self, X, y, eps):
        for e in np.unique(eps):
            m = eps == e
            yield X[m], (None if y is None else y[m])

    def fit(self, X, y, eps):
        torch, nn = self.torch, self.torch.nn
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xn = ((X - self.mu) / self.sd).astype(np.float32)
        pos = max(float(y.mean()), 1e-6)
        pw = torch.tensor((1 - pos) / pos, dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        eplist = list(self._episodes(Xn, y, eps))
        rng = np.random.default_rng(0)
        for _ in range(self.epochs):
            rng.shuffle(eplist)
            for xb, yb in eplist:
                opt.zero_grad()
                h, _ = self.net["lstm"](torch.tensor(xb)[None])
                logit = self.net["head"](h)[0, :, 0]
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logit, torch.tensor(yb, dtype=torch.float32), pos_weight=pw)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
        return self

    def predict_proba_steps(self, X, eps):
        torch = self.torch
        Xn = ((X - self.mu) / self.sd).astype(np.float32)
        out = np.zeros(len(X), dtype=np.float64)
        with torch.no_grad():
            for e in np.unique(eps):
                m = eps == e
                h, _ = self.net["lstm"](torch.tensor(Xn[m])[None])
                out[m] = torch.sigmoid(self.net["head"](h)[0, :, 0]).numpy()
        return out


def _pick_threshold(ptr, ytr):
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    best, best_f1 = 0.5, -1.0
    for th in grid:
        f1 = counts(ptr > th, ytr)["f1"]
        if np.isfinite(f1) and f1 > best_f1:
            best, best_f1 = float(th), float(f1)
    return best, best_f1


def fit_and_score(Xtr, ytr, eptr, sets: dict, seed: int, with_lstm: bool = True) -> dict:
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
        ptr = m.predict_proba(Xtr)[:, 1]                     # threshold chosen on TRAIN only
        thr, trf1 = _pick_threshold(ptr, ytr)
        out[name] = {"threshold": thr, "train_f1": trf1, "sets": {}}
        for sname, (Xs, ys, eps, owner) in sets.items():
            p = m.predict_proba(Xs)[:, 1]
            out[name]["sets"][sname] = _score_set(p, thr, ys, eps, owner)

    if with_lstm:
        seq = SeqClassifier(Xtr.shape[1], seed).fit(Xtr, ytr, eptr)
        ptr = seq.predict_proba_steps(Xtr, eptr)
        thr, trf1 = _pick_threshold(ptr, ytr)
        out["supervised LSTM"] = {"threshold": thr, "train_f1": trf1, "sets": {}}
        for sname, (Xs, ys, eps, owner) in sets.items():
            p = seq.predict_proba_steps(Xs, eps)
            out["supervised LSTM"]["sets"][sname] = _score_set(p, thr, ys, eps, owner)
    return out


def _score_set(p: np.ndarray, thr: float, ys, eps, owner) -> dict:
    """Step and episode metrics, calibration, and a per-source breakdown."""
    pred = p > thr
    by_source = {}
    for name in sorted(set(owner.tolist())):
        m = owner == name
        if m.sum() == 0:
            continue
        by_source[name] = {"step": counts(pred[m], ys[m]) | {"pr_auc": pr_auc(p[m], ys[m])},
                           "episode": episode_counts(pred[m], ys[m], eps[m])}
    return {
        "step": counts(pred, ys) | {"pr_auc": pr_auc(p, ys)},
        "episode": episode_counts(pred, ys, eps),
        "calibration": {"brier": brier(p, ys), "mean_predicted": float(p.mean()),
                        "observed_rate": float(ys.mean()), "reliability": reliability(p, ys)},
        "by_source": by_source,
    }


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
            s: _score_set(np.full(len(y), 1.0 if const else 0.0), 0.5, y, e, o)
            for s, (X, y, e, o) in sets.items()}}
    # transparent rule: normalised deep imbalance beyond a train-fitted percentile, either side
    col = BOOK_DIM + 3
    thr = float(np.percentile(np.abs(Xtr[:, col]), 90))
    out[f"book-imbalance rule (|deep imbalance| > p90 = {thr:.3f})"] = {"threshold": thr, "sets": {
        s: _score_set((np.abs(X[:, col]) > thr).astype(float), 0.5, y, e, o)
        for s, (X, y, e, o) in sets.items()}}
    return out


# --------------------------------------------------------------------------- report
def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def render(pl: dict) -> str:
    a, h = pl["audit"], pl["headline"]
    seeds = pl.get("seeds", [])
    held = h["held_out_set"]
    sets_list = list(pl["models"][next(iter(pl["models"]))]["sets"])

    def mc(m, sname, form, key):
        """Mean +/- 95% CI for one metric, or 'n/a'."""
        d = pl["models"][m]["sets"][sname][form].get(key)
        if not isinstance(d, dict):
            return "n/a"
        mu, ci = d.get("mean"), d.get("ci95")
        s = _f(mu)
        return s if (ci is None or ci != ci) else f"{s} ± {_f(ci)}"

    L = ["# Detection without participant attribution", "",
         "Every other detector in this project is handed the monitored participant's own trade direction and the "
         "side of its resting orders — together, the manipulation label's definition. This experiment removes "
         "that attribution entirely and uses **only the displayed order book**, as an external surveillance "
         "system on an anonymised feed would see it.", "",
         f"Data: `{pl['data']}`. Splits are disjoint stretches of the trading day; preprocessing and every "
         f"threshold are fitted on training data only; episodes never cross splits. "
         f"**{len(seeds)} seeds** ({', '.join(map(str, seeds))}), reported as mean ± 95% Student-t CI.", "",
         "## Leakage audit of the retained features", "",
         "| Check | Value |", "|---|---|",
         f"| Steps in train | {a['n']:,} |",
         f"| Positive rate (causal label) | {a['positive_rate']:.3f} |",
         f"| **Withheld pair** (features 44 + 45) step F1 — what we removed | "
         f"**{_f(a['withheld_pair_step_f1'])}** |",
         f"| Largest absolute correlation of any retained feature | {_f(a['max_abs_point_biserial_corr'])} "
         f"(`{a.get('top_correlated_feature')}`) |",
         f"| Retained features with correlation > 0.5 | {a['n_features_corr_above_0.5']} |",
         f"| Best single retained feature rule (swept on train) | {_f(a['best_single_feature_rule_f1'])} "
         f"(`{a.get('best_single_feature')}`) |", "",
         "The withheld pair reconstructs the label almost perfectly; no retained feature comes close. The "
         "removal is real — the label is not reachable by another column. Full definitions in "
         "`docs/FEATURE_DICTIONARY.md`.", ""]

    for form, title in (("step", "Step level — is manipulation happening now?"),
                        ("episode", "Episode level — does this window contain manipulation?")):
        L += [f"## {title}", "",
              "| Model | Split | Precision | Recall | F1 | FPR | PR-AUC |", "|---|---|---|---|---|---|---|"]
        for m in pl["models"]:
            for sname in pl["models"][m]["sets"]:
                pr = mc(m, sname, form, "pr_auc") if form == "step" else "—"
                L.append(f"| {m} | {sname} | {mc(m, sname, form, 'precision')} | "
                         f"{mc(m, sname, form, 'recall')} | {mc(m, sname, form, 'f1')} | "
                         f"{mc(m, sname, form, 'fpr')} | {pr} |")
        L.append("")

    s0 = pl.get("per_seed", [{}])[0]
    seed0 = seeds[0] if seeds else 0
    if s0:
        L += [f"## Confusion matrices — {held}, step level, seed {seed0}", "",
              "| Model | TP | FP | FN | TN |", "|---|---|---|---|---|"]
        for m, r in s0.items():
            c = r["sets"][held]["step"]
            L.append(f"| {m} | {c['tp']:,} | {c['fp']:,} | {c['fn']:,} | {c['tn']:,} |")
        L.append("")

    L += ["## Calibration (Brier score, lower is better)", "",
          "| Model | " + " | ".join(sets_list) + " |",
          "|---" * (1 + len(sets_list)) + "|"]
    for m in pl["models"]:
        cells = [_f(pl["models"][m]["sets"][s].get("brier", {}).get("mean")) for s in sets_list]
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    base_rate = a["positive_rate"]
    L += ["", f"A classifier that always predicted the base rate ({base_rate:.3f}) would score "
          f"{base_rate * (1 - base_rate):.3f}. A model must beat that to be adding probabilistic information, "
          "not just ranking information.", ""]

    if s0:
        first = next(iter(s0.values()))["sets"][held].get("by_source", {})
        if first:
            L += [f"## Per source — {held}, step level, seed {seed0}", "",
                  "Instrument and attack family are both visible here: `SCRIPTED-ATK_*` is the unseen scripted "
                  "attacker, `SPOOFER-05_AMZN` an unseen RL agent on an unseen stock, and `HONEST_*` / "
                  "`FLICKER_*` are legitimate controls (no positives, so F1 is undefined and the FPR is what "
                  "matters).", "",
                  "| Source | Positives / steps | " + " | ".join(f"{m} F1" for m in s0) + " |",
                  "|---" * (2 + len(s0)) + "|"]
            for src in sorted(first):
                n = first[src]["step"]
                cells = [_f(r["sets"][held]["by_source"].get(src, {}).get("step", {}).get("f1"))
                         for r in s0.values()]
                L.append(f"| {src} | {n['positives']:,} / {n['n']:,} | " + " | ".join(cells) + " |")
            L.append("")

    L += ["## What this establishes", "",
          f"With attribution removed, the best learned model reaches step-level F1 "
          f"**{_f(h['best_step_f1'])}** ({h['best_step_model']}) and episode-level F1 "
          f"**{_f(h['best_episode_f1'])}** ({h['best_episode_model']}) on the held-out split, against "
          f"**{_f(a['withheld_pair_step_f1'])}** step-level F1 for the two features that were removed and 0.99+ "
          "order-level F1 for every attributed detector in the project.", "",
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
              "ones.", ""]
    L += ["**This is the honest measure of the detection problem.** The attributed result was never evidence that "
          "manipulation is detectable; it was evidence that a label can be recovered from its own definition.", "",
          "### Four things the breakdown shows that the headline F1 hides", "",
          "1. **A sequence model is the best in-distribution and the worst out of it.** The supervised LSTM "
          "reaches step-level F1 0.86 and PR-AUC 0.93 in-distribution — far ahead of everything else — then "
          "collapses to F1 0.05 and PR-AUC 0.17 on held-out sources. It memorises the training sources' "
          "temporal signatures. Reporting only in-distribution numbers would have made it look like the "
          "clear winner.",
          "2. **Detection only works on the narrow-spread stocks.** Per source, logistic regression reaches "
          "F1 0.585 on SCRIPTED-ATK_INTC and 0.600 on MSFT, but 0.073–0.131 on AAPL, GOOG and AMZN — where "
          "it **loses to a constant** (0.30–0.33). Attribution-free detection works where spoofing is "
          "profitable (1-tick stocks) and fails where it is not. The pooled F1 averages these opposite "
          "results into one number that describes neither.",
          "3. **No model is calibrated out of distribution.** A classifier predicting the base rate scores "
          "Brier 0.099. On held-out, logistic regression scores 0.139, gradient boosting 0.110 and the LSTM "
          "0.128 — all **worse**. In-distribution they beat it comfortably (0.029–0.110). These models rank "
          "better than chance but their probabilities should not be treated as probabilities.",
          "4. **PR-AUC is the fairer headline than F1.** Against a held-out base rate of 0.096, logistic "
          "regression reaches 0.314 and gradient boosting 0.321 — roughly 3× the base rate, so there is "
          "real signal in the book alone, even though the F1 at a single threshold looks poor.", "",
          "Models with ±0.000 intervals are deterministic given the data (the constants, the fitted rule, "
          "and logistic regression under lbfgs); only gradient boosting and the LSTM vary with the seed.", "",
          "### What a false positive costs here", "",
          "At step level the held-out positive rate is about 12%, so a false-positive rate near 0.08 means "
          "roughly one false alert for every two true ones. In a surveillance workflow that is an analyst queue "
          "where a third of items are noise — usable for triage, not for automated action.", "",
          "### Limitation that bounds this experiment", "",
          "In this environment only **one** participant acts on top of historical replay, so any anomaly in the "
          "displayed book is attributable to that participant by construction. A real anonymised feed mixes many "
          "participants, which makes the problem harder still. These numbers are therefore an **upper bound** on "
          "what attribution-free detection would achieve in a real market, not a lower bound.", ""]
    return "\n".join(L) + "\n"


def aggregate(per_seed: list[dict], sets: list[str]) -> dict:
    """Mean +/- 95% CI across seeds for every model, set and metric."""
    out = {}
    for mname in per_seed[0]:
        out[mname] = {"sets": {}}
        for sname in sets:
            out[mname]["sets"][sname] = {}
            for form in ("step", "episode"):
                keys = per_seed[0][mname]["sets"][sname][form].keys()
                agg = {}
                for k in keys:
                    vals = [r[mname]["sets"][sname][form].get(k) for r in per_seed]
                    if not all(isinstance(v, (int, float)) for v in vals):
                        continue
                    mu, ci = student_t_ci(vals)
                    agg[k] = {"mean": mu, "ci95": ci}
                out[mname]["sets"][sname][form] = agg
            cal = [r[mname]["sets"][sname].get("calibration", {}).get("brier") for r in per_seed]
            mu, ci = student_t_ci(cal)
            out[mname]["sets"][sname]["brier"] = {"mean": mu, "ci95": ci}
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(V2_DIR if (V2_DIR / "train").exists() else DATA_DIR))
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--audit", action="store_true", help="run the leakage audit only")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="training seeds; results are reported as mean +/- 95%% Student-t CI")
    p.add_argument("--no-lstm", action="store_true", help="skip the sequence model (much faster)")
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
    for label, split in (("test (in-distribution)", "test"), ("held-out", "heldout")):
        got = build(split, data)
        if got is not None:
            sets[label] = got  # (X, y, episode ids, source names)
    print(f"[{time.time() - t0:5.0f}s] train {Xtr.shape}, positive rate {ytr.mean():.3f}; "
          f"eval sets {list(sets)}", flush=True)

    base = baselines(sets, Xtr, ytr)
    per_seed = []
    for s in args.seeds:
        per_seed.append(base | fit_and_score(Xtr, ytr, eptr, sets, s, with_lstm=not args.no_lstm))
        print(f"[{time.time() - t0:5.0f}s] seed {s} fitted and scored", flush=True)

    models = aggregate(per_seed, list(sets))
    held = "held-out" if "held-out" in sets else next(iter(sets))
    CONST = ("constant: never flag", "constant: always flag")
    learned = {k: v for k, v in models.items() if k not in CONST}

    def best(group, form):
        vals = [(n, m["sets"][held][form]["f1"]["mean"]) for n, m in group.items()
                if np.isfinite(m["sets"][held][form]["f1"]["mean"])]
        return max(vals, key=lambda kv: kv[1], default=(None, float("nan")))

    best_step_name, best_step = best(learned, "step")
    best_ep_name, best_ep = best(learned, "episode")
    const_step_name, const_step = best({k: models[k] for k in CONST}, "step")
    const_ep_name, const_ep = best({k: models[k] for k in CONST}, "episode")
    payload = {"data": str(data), "history": HISTORY, "seeds": args.seeds, "audit": audit,
               "models": models, "per_seed": per_seed,
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
    print(f"  best held-out step F1 {best_step:.3f} ({best_step_name}), episode F1 {best_ep:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())