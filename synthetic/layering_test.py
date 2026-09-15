"""Pre-registered layering test in the synthetic market (see README, "Pre-registered layering test").

This script, the attacker (synthetic/layering.py) and the hypothesis were committed before any LAYER-ATK data or
result existed. Run it once; do not change the Watchdog, the rule, thresholds or features after seeing its output.

Usage:
    python synthetic/layering_test.py

Procedure (fixed in advance):
  * The calibrated synthetic market (configs/synthetic_calibration.json), 10 market events per step.
  * 30 episodes of LAYER-ATK (4 layers, 10x touch depth in total) and 30 of FLICKER (negatives), with imbalance
    followers reacting to the agent's orders.
  * The main Watchdog (trained on replay data only, with its saved observation normalisation) and the rule tuned
    on the replay train split, scored on the same recorded orders.
  * Hypothesis: the Watchdog's order-level F1 on LAYER-ATK + FLICKER is at least the rule's.
    Supported if Watchdog F1 >= rule F1; not supported otherwise.
  * Context only (not used for the verdict): LAYER-ATK PnL with reactive vs blind followers, recall by layer depth.

Writes results/layering_test.md and results/layering_test.json.
"""
from __future__ import annotations

import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from env.lob_env import EnvConfig  # noqa: E402
from evaluation.agents import Flicker  # noqa: E402
from evaluation.baseline_detector import tune  # noqa: E402
from evaluation.rollout import record_episode  # noqa: E402
from synthetic.layering import LayerSpoof  # noqa: E402
from synthetic.spoof_impact import load_params  # noqa: E402
from synthetic.synthetic_env import SyntheticSpoofEnv  # noqa: E402
from watchdog.dataset import _orders_to_array, load_split, orders_from_array  # noqa: E402
from watchdog.evaluate_watchdog import combine, order_flags, score_source  # noqa: E402
from watchdog.watchdog_env import load_obs_norm  # noqa: E402

EPISODES = 30
SEED_BASE = 800_000


def ci95(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(x.mean()), (float(1.96 * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan"))


def run(name, policy, cfg, params, blind, features):
    import torch
    base = SEED_BASE + zlib.crc32(name.encode()) % 10_000
    env = SyntheticSpoofEnv(EnvConfig(**cfg), params, warmup_events=6_000, blind_followers=blind)
    rng = np.random.default_rng(base)
    torch.manual_seed(base)
    return [record_episode(env, policy, rng, base + 97 * j, features=features) for j in range(EPISODES)]


def as_source(eps) -> dict:
    lengths = np.array([len(e["y"]) for e in eps])
    return {"X": np.concatenate([e["X"] for e in eps]), "y": np.concatenate([e["y"] for e in eps]),
            "ep_offsets": np.concatenate([[0], np.cumsum(lengths)]),
            "orders": _orders_to_array([e["orders"] for e in eps])}


def main() -> int:
    from sb3_contrib import RecurrentPPO
    from train_spoofer import DECISION_ENV

    t0 = time.time()
    params = load_params()
    wd_path = ROOT / "checkpoints" / "WATCHDOG" / "main" / "model.zip"
    model = RecurrentPPO.load(wd_path, device="cpu")
    obs_norm = load_obs_norm(wd_path.parent)
    train = load_split("train")
    det, _ = tune([o for s in train.values() for _, o in orders_from_array(s["orders"])])

    layer_eps = run("LAYER-ATK", LayerSpoof(), DECISION_ENV, params, blind=False, features=True)
    flicker_eps = run("FLICKER", Flicker(), DECISION_ENV, params, blind=False, features=True)
    layer_blind = run("LAYER-ATK", LayerSpoof(), DECISION_ENV, params, blind=True, features=False)
    print(f"[{time.time() - t0:5.0f}s] episodes recorded", flush=True)

    layer_src, flicker_src = as_source(layer_eps), as_source(flicker_eps)
    scored_layer = score_source(model, layer_src, det, obs_norm)
    scored_flicker = score_source(model, flicker_src, det, obs_norm)
    pooled = combine([scored_layer, scored_flicker])
    w_f1, r_f1 = pooled["watchdog_orders"]["f1"], pooled["rule_orders"]["f1"]
    verdict = "supported" if w_f1 >= r_f1 else "not supported"

    # context: recall by layer. LayerSpoof places the best-price layer first and each deeper layer after it, so
    # within one placement (same episode, step and side) the order-id rank equals the layer depth (0 = best price).
    orders = orders_from_array(layer_src["orders"])
    decisions, _ = order_flags(orders, [np.asarray(scored_layer["step_flags"][a:b])
                                        for a, b in zip(layer_src["ep_offsets"][:-1], layer_src["ep_offsets"][1:])])
    groups: dict[tuple[int, int, int], list[int]] = {}
    for i, (ep, o) in enumerate(orders):
        groups.setdefault((ep, o.placed_step, o.side), []).append(i)
    depth_of = {}
    for idx in groups.values():
        for rank, i in enumerate(sorted(idx, key=lambda j: orders[j][1].order_id)):
            depth_of[i] = rank
    by_depth = {}
    for i, (ep, o) in enumerate(orders):
        if o.manipulative:
            by_depth.setdefault(depth_of.get(i, -1), []).append((decisions[i], det.flags(o)))
    recall_by_depth = {int(k): {"n": len(v), "watchdog": float(np.mean([a for a, _ in v])), "rule": float(np.mean([b for _, b in v]))}
                       for k, v in sorted(by_depth.items())}

    reactive_mean, reactive_ci = ci95([e["pnl"] for e in layer_eps])
    blind_mean, blind_ci = ci95([e["pnl"] for e in layer_blind])
    gain_mean, gain_ci = ci95(np.array([e["pnl"] for e in layer_eps]) - np.array([e["pnl"] for e in layer_blind]))
    payload = {
        "hypothesis": "Watchdog order-level F1 on LAYER-ATK + FLICKER >= rule F1",
        "verdict": verdict, "watchdog_f1": w_f1, "rule_f1": r_f1,
        "pooled": pooled, "layer_only": combine([scored_layer]), "flicker_only": combine([scored_flicker]),
        "recall_by_layer_depth": recall_by_depth,
        "layer_pnl": {"reactive_mean": reactive_mean, "reactive_ci95": reactive_ci, "blind_mean": blind_mean,
                      "blind_ci95": blind_ci, "gain_mean": gain_mean, "gain_ci95": gain_ci},
        "rule": {"n_mult": det.n_mult, "m_events": det.m_events}, "episodes": EPISODES,
        "runtime_s": round(time.time() - t0, 1),
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "layering_test.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out / "layering_test.md").write_text(render(payload), encoding="utf-8")
    print(f"verdict: {verdict} (Watchdog F1 {w_f1:.3f}, rule F1 {r_f1:.3f})")
    print(f"wrote {out / 'layering_test.md'}")
    return 0


def _f(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{nd}f}"


def render(pl: dict) -> str:
    r = pl["rule"]
    L = ["# Pre-registered layering test (synthetic market)", "",
         "Defined and committed before any of its data or results existed; nothing was changed after seeing them.", "",
         f"**Hypothesis:** {pl['hypothesis']}.", "",
         f"**Result: {pl['verdict']}.** Watchdog F1 = {_f(pl['watchdog_f1'], 3)}, rule F1 = {_f(pl['rule_f1'], 3)}.", "",
         f"Rule tuned on the replay train split: size ≥ {r['n_mult']}× depth, cancelled within {r['m_events']} events. "
         f"{pl['episodes']} episodes each of LAYER-ATK (4 layers) and FLICKER in the calibrated synthetic market.", "",
         "| Bucket | Manipulative / orders | Watchdog P | Watchdog R | Watchdog F1 | Watchdog FPR | Rule P | Rule R | Rule F1 | Rule FPR |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, key in (("LAYER-ATK + FLICKER", "pooled"), ("LAYER-ATK only", "layer_only"), ("FLICKER only", "flicker_only")):
        b = pl[key]
        w, rl = b["watchdog_orders"], b["rule_orders"]
        L.append(f"| {name} | {w['positives']} / {w['n']} | {_f(w['precision'])} | {_f(w['recall'])} | {_f(w['f1'])} | "
                 f"{_f(w['fpr'])} | {_f(rl['precision'])} | {_f(rl['recall'])} | {_f(rl['f1'])} | {_f(rl['fpr'])} |")
    L += ["", "Context (not used for the verdict).", "",
          "| Layer (0 = first placed) | Manipulative orders | Watchdog recall | Rule recall |", "|---|---|---|---|"]
    for k, v in pl["recall_by_layer_depth"].items():
        L.append(f"| {k} | {v['n']} | {_f(v['watchdog'])} | {_f(v['rule'])} |")
    p = pl["layer_pnl"]
    L += ["", f"LAYER-ATK PnL per episode: reactive {_f(p['reactive_mean'], 1)} ± {_f(p['reactive_ci95'], 1)}, "
          f"blind followers {_f(p['blind_mean'], 1)} ± {_f(p['blind_ci95'], 1)}, "
          f"manipulation gain {_f(p['gain_mean'], 1)} ± {_f(p['gain_ci95'], 1)}."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
