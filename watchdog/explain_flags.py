"""Concrete explanations of individual Watchdog decisions (execution plan, Phase 8; CP-6).

Usage:
    python watchdog/explain_flags.py [--tag cov]

The Watchdog has no attention layer, so each decision is explained by occlusion, consistent with
watchdog/feature_ablation.py: a feature (group) is set to its training mean (0 after the saved normalisation) and the
change in the policy's flag probability at the explained step is measured, two ways:
  this step   only the explained step is occluded (what the Watchdog reads now)
  history     every earlier step of the episode is occluded (what the LSTM carries from the past)
plus one counterfactual: the sign of this step's trade direction is flipped (same trade, other side).

Four decisions are chosen by fixed rules (the first match in sorted source / episode / order order, nothing
hand-picked), from data that is already on disk and already evaluated:
  A  true positive, sparse RL Spoofer: SPOOFER-03 GOOG (test split), first opposite trade 3+ steps after placement
  B  true positive, pre-registered REVERSAL-ATK: an order placed on the step right after the previous leg's cancel
  C  false positive, legitimate boundary case REVERT (opposite trades just after its own cancel)
  D  true negative, ACCUM: a same-side trade while its own large order rests (training-only source, test split)

Writes results/flag_explanations.md and results/flag_explanations.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.rollout import BOOK_DIM  # noqa: E402
from watchdog.dataset import EXT_DATA_DIR, TRADE_COL, load_split, orders_from_array  # noqa: E402
from watchdog.watchdog_env import load_obs_norm, prepare_obs  # noqa: E402

PARTICIPANT = ["resting size", "oldest order age", "placed this step", "cancelled this step", "trade direction",
               "resting side imbalance"]
GROUPS = {name: [BOOK_DIM + i] for i, name in enumerate(PARTICIPANT)}
GROUPS["all 40 book features"] = list(range(BOOK_DIM))


def flag_probs(model, Xn: np.ndarray) -> np.ndarray:
    """P(flag) at every step of one episode, the recurrent state carried from the episode start."""
    lstm = model.policy.lstm_actor
    state = (torch.zeros(lstm.num_layers, 1, lstm.hidden_size), torch.zeros(lstm.num_layers, 1, lstm.hidden_size))
    starts = torch.zeros(len(Xn))
    starts[0] = 1.0
    with torch.no_grad():
        dist, _ = model.policy.get_distribution(torch.as_tensor(Xn, dtype=torch.float32), state, starts)
    return dist.distribution.probs[:, 1].numpy()


def episodes(src: dict):
    off = src["ep_offsets"]
    for ep, (a, b) in enumerate(zip(off[:-1], off[1:])):
        yield ep, src["X"][a:b]


def find(model, norm, src: dict, rule):
    """First (episode, order, explained step, X, probs) for which rule(order, X, probs) returns a step."""
    orders = sorted(orders_from_array(src["orders"]), key=lambda eo: (eo[0], eo[1].order_id))
    by_ep = dict(episodes(src))
    cache = {}
    for ep, o in orders:
        X = by_ep[ep]
        if ep not in cache:
            cache[ep] = flag_probs(model, prepare_obs(X, norm))
        step = rule(o, X, cache[ep], orders)
        if step is not None:
            return ep, o, step, X, cache[ep]
    return None


def rule_sparse_tp(o, X, p, orders):
    acts = np.flatnonzero(X[o.placed_step:o.removed_step, TRADE_COL] == -o.side)
    if o.manipulative and acts.size and acts[0] >= 3:
        flagged = np.flatnonzero(p[o.placed_step:o.removed_step] > 0.5)
        return int(o.placed_step + flagged[0]) if flagged.size else None
    return None


def rule_reversal_tp(o, X, p, orders):
    prev = [q for e, q in orders if q.removed_step == o.placed_step - 1 and q.side == -o.side]
    if o.manipulative and prev:
        flagged = np.flatnonzero(p[o.placed_step:o.removed_step] > 0.5)
        return int(o.placed_step + flagged[0]) if flagged.size else None
    return None


def rule_boundary_fp(o, X, p, orders):
    lo, hi = o.placed_step, min(o.removed_step + 4, len(p))
    flagged = np.flatnonzero(p[lo:hi] > 0.5)
    return int(lo + flagged[0]) if not o.manipulative and flagged.size else None


def rule_sameside_tn(o, X, p, orders):
    same = np.flatnonzero(X[o.placed_step:o.removed_step, TRADE_COL] == o.side)
    if not o.manipulative and same.size and not (p[o.placed_step:o.removed_step] > 0.5).any():
        return int(o.placed_step + same[0])
    return None


def attribute(model, norm, X: np.ndarray, step: int, base: float) -> dict:
    Xn = prepare_obs(X, norm)
    out = {}
    for name, idx in GROUPS.items():
        for mode in ("this step", "history"):
            Xo = Xn.copy()
            if mode == "this step":
                Xo[step, idx] = 0.0
            elif step > 0:
                Xo[:step, idx] = 0.0
            out[f"{name} | {mode}"] = float(flag_probs(model, Xo)[step] - base)
    Xf = X.copy()
    Xf[step, TRADE_COL] *= -1.0
    out["counterfactual: trade on the other side"] = float(flag_probs(model, prepare_obs(Xf, norm))[step] - base)
    return out


def timeline(X: np.ndarray, p: np.ndarray, o, step: int) -> list[dict]:
    lo, hi = max(o.placed_step - 1, 0), min(o.removed_step + 3, len(p))
    rows = [{"step": s, "rel": s - o.placed_step, "resting": float(X[s, BOOK_DIM]), "age": float(X[s, BOOK_DIM + 1]),
             "placed": float(X[s, BOOK_DIM + 2]), "cancelled": float(X[s, BOOK_DIM + 3]),
             "trade": int(X[s, TRADE_COL]), "imbalance": float(X[s, BOOK_DIM + 5]), "p_flag": float(p[s]),
             "explained": s == step} for s in range(lo, hi)]
    if len(rows) > 24:   # keep the placement, every trade or cancel, the explained step and its neighbours
        keep = {0, len(rows) - 1} | {i for i, r in enumerate(rows) for d in (-1, 0, 1)
                                     if 0 <= i + d < len(rows) and (rows[i + d]["trade"] or rows[i + d]["cancelled"]
                                                                   or rows[i + d]["placed"] or rows[i + d]["explained"])}
        rows = [r for i, r in enumerate(rows) if i in keep]
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="cov")
    args = p.parse_args()
    from sb3_contrib import RecurrentPPO
    path = ROOT / "checkpoints" / "WATCHDOG" / args.tag / "model.zip"
    model, norm = RecurrentPPO.load(path, device="cpu"), load_obs_norm(path.parent)
    test, rev, ext = load_split("test"), load_split("reversal"), load_split("test", EXT_DATA_DIR)

    cases = [
        ("A", "True positive: sparse RL Spoofer", test, ["SPOOFER-03_GOOG"], rule_sparse_tp),
        ("B", "True positive: pre-registered REVERSAL-ATK, the reversed leg", rev,
         sorted(n for n in rev if n.startswith("REVERSAL")), rule_reversal_tp),
        ("C", "False positive: legitimate REVERT (opposite trades just after its own cancel)", rev,
         sorted(n for n in rev if n.startswith("REVERT")), rule_boundary_fp),
        ("D", "True negative: ACCUM trades on the same side as its resting order", ext,
         sorted(n for n in ext if n.startswith("ACCUM")), rule_sameside_tn),
    ]
    out = []
    for key, title, split, names, rule in cases:
        for name in names:
            hit = find(model, norm, split[name], rule)
            if hit:
                break
        if not hit:
            out.append({"case": key, "title": title, "found": False})
            continue
        ep, o, step, X, probs = hit
        out.append({"case": key, "title": title, "found": True, "source": name, "episode": ep,
                    "order": {"id": o.order_id, "side": "bid (buy)" if o.side > 0 else "ask (sell)",
                              "placed": o.placed_step, "removed": o.removed_step, "removed_by": o.removed_by,
                              "opposite_trades": o.opposite_trades, "manipulative": o.manipulative},
                    "explained_step": step, "p_flag": float(probs[step]),
                    "timeline": timeline(X, probs, o, step),
                    "attribution": attribute(model, norm, X, step, float(probs[step]))})
        print(f"case {key}: {name} episode {ep} order {o.order_id} step {step} p_flag={probs[step]:.3f}", flush=True)

    payload = {"watchdog": f"checkpoints/WATCHDOG/{args.tag}", "cases": out}
    res = ROOT / "results"
    (res / "flag_explanations.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (res / "flag_explanations.md").write_text(render(payload), encoding="utf-8")
    print(f"wrote {res / 'flag_explanations.md'}")
    return 0


def _top(attr: dict, n: int = 3) -> list[tuple[str, float]]:
    occl = {k: v for k, v in attr.items() if not k.startswith("counterfactual")}
    return sorted(occl.items(), key=lambda kv: -abs(kv[1]))[:n]


def render(pl: dict) -> str:
    L = ["# Watchdog decisions explained", "",
         f"Watchdog `{pl['watchdog']}`. Each decision is explained by occlusion: a feature is set to its training mean "
         "(0 after normalisation) either at the explained step only or at every earlier step, and the change in the "
         "flag probability at the explained step is shown. The counterfactual flips the side of this step's trade. "
         "Cases are picked by fixed rules (first match), not by hand. Participant features are as in "
         "`evaluation/rollout.py`: resting size and placed / cancelled are log1p multiples of touch depth, age is "
         "steps / 20, trade is +1 buy / −1 sell, imbalance is signed (bid minus ask).", ""]
    for c in pl["cases"]:
        L += [f"## {c['case']}. {c['title']}", ""]
        if not c["found"]:
            L += ["No matching decision found.", ""]
            continue
        o = c["order"]
        L += [f"`{c['source']}`, episode {c['episode']}: large {o['side']} order {o['id']}, placed at step {o['placed']}, "
              f"removed at step {o['removed']} ({o['removed_by']}), {o['opposite_trades']} opposite trade(s) while "
              f"resting, label {'manipulative' if o['manipulative'] else 'legitimate'}. Explained step "
              f"**{c['explained_step']}**, flag probability **{c['p_flag']:.3f}**.", "",
              "| Step | From placement | Resting size | Age | Placed | Cancelled | Trade | Imbalance | P(flag) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in c["timeline"]:
            mark = "**" if r["explained"] else ""
            trade = {1: "buy", -1: "sell", 0: ""}[r["trade"]]
            L.append(f"| {mark}{r['step']}{mark} | {r['rel']:+d} | {r['resting']:.2f} | {r['age']:.2f} | {r['placed']:.2f} | "
                     f"{r['cancelled']:.2f} | {trade} | {r['imbalance']:+.2f} | {mark}{r['p_flag']:.3f}{mark} |")
        L += ["", "| Occluded | Change in P(flag) |", "|---|---|"]
        for k, v in sorted(c["attribution"].items(), key=lambda kv: -abs(kv[1])):
            L.append(f"| {k} | {v:+.3f} |")
        top = _top(c["attribution"])
        cf = c["attribution"]["counterfactual: trade on the other side"]
        L += ["", "Largest effects: " + "; ".join(f"{k} ({v:+.3f})" for k, v in top)
              + f". Same trade on the other side: {cf:+.3f}.", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
