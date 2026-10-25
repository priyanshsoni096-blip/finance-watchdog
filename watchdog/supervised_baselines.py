"""Baselines a reviewer will ask for next to the CP-7 Watchdog: is RL needed, and how does it compare with the rule
that defines the label?

Usage:
    python watchdog/supervised_baselines.py train      # fit on the train splits, write test-split comparison
    python watchdog/supervised_baselines.py final      # score once on every held-out dataset (after committing)

Detectors, all fed the same 46-dim surveillance observation:
  LABEL-RULE  flag from the first step on which the owner trades on the side opposite its own resting large order
              until that order is removed (watchdog.dataset.causal_labels). This is the definition of the proxy label,
              written as a rule, so it is perfect on order-level labels by construction; it is the ceiling, not a
              competitor.
  LOGREG      logistic regression on [this step, mean of the previous 10 steps], plain (unweighted) cross-entropy.
  SUP-LSTM    an LSTM classifier the size of the Watchdog's policy (LSTM 64 -> 128 tanh -> 1), trained by
              cross-entropy on whole episodes, sources sampled uniformly as in WatchdogEnv.
Supervised models are trained on exactly the CP-7 Watchdog's data: the original train split plus data/watchdog_ext,
causal labels, observations clipped to +-25 and standardised with training statistics. With the Watchdog's reward
(+1 / -1 / -2 / 0) the cost-optimal decision is flag iff P(manipulative) > 0.5, so both use a 0.5 threshold.

`train` writes checkpoints/SUPERVISED/*.pt and results/baselines_dev.md (test splits only). `final` writes
results/baselines_final.md / .json: order-level recall, legitimate FPR and pooled F1 on the original test split, the
held-out split, and the saved LATEBURST-ATK, DRIP-ATK and REVERSAL-ATK data, next to the CP-7 Watchdog (`cov`, 3
seeds) and the size + fast-cancel rule.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.baseline_detector import tune  # noqa: E402
from evaluation.rollout import OBS_DIM  # noqa: E402
from watchdog.dataset import EXT_DATA_DIR, causal_labels, load_split, orders_from_array  # noqa: E402
from watchdog.evaluate_watchdog import combine, counts, order_flags, score_source  # noqa: E402
from watchdog.multiseed_summary import mean_ci95  # noqa: E402
from watchdog.watchdog_env import OBS_BOUND, load_obs_norm  # noqa: E402

OUT = ROOT / "checkpoints" / "SUPERVISED"
HISTORY = 10
SEEDS = (7, 11, 23)
WATCHDOG_TAGS = ("cov", "cov_s2", "cov_s3")


# ------------------------------------------------------------------ data
def train_sources() -> dict[str, dict]:
    src = {**load_split("train"), **load_split("train", EXT_DATA_DIR)}
    return {n: {**s, "y": causal_labels(s)} for n, s in src.items()}


def episodes(src: dict):
    off = src["ep_offsets"]
    return [(src["X"][a:b], src["y"][a:b]) for a, b in zip(off[:-1], off[1:]) if b > a]


def history_features(X: np.ndarray) -> np.ndarray:
    """[x_t, mean of x_{t-HISTORY..t-1}] for one episode (zeros before the first step)."""
    csum = np.cumsum(np.vstack([np.zeros((1, X.shape[1]), np.float32), X]), axis=0)
    idx = np.arange(len(X))
    lo = np.maximum(idx - HISTORY, 0)
    prev = (csum[idx] - csum[lo]) / np.maximum(idx - lo, 1)[:, None]
    return np.hstack([X, prev]).astype(np.float32)


# ---------------------------------------------------------------- models
class SupLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(OBS_DIM, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, x, state=None):
        h, state = self.lstm(x, state)
        return self.head(h).squeeze(-1), state


class Standardiser:
    def __init__(self, mean, std):
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)

    def __call__(self, X):
        return ((np.clip(X, -OBS_BOUND, OBS_BOUND) - self.mean) / self.std).astype(np.float32)


def fit_logreg(sources, std: Standardiser, seed: int):
    torch.manual_seed(seed)
    feats = [history_features(std(X)) for s in sources.values() for X, _ in episodes(s)]
    ys = [y for s in sources.values() for _, y in episodes(s)]
    X, y = torch.tensor(np.vstack(feats)), torch.tensor(np.concatenate(ys), dtype=torch.float32)
    lin = nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(lin(X).squeeze(-1), y)
        loss.backward()
        opt.step()
    return lin, float(loss)


def fit_lstm(sources, std: Standardiser, seed: int, updates: int = 3000, batch: int = 32):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    names = sorted(sources)
    eps = {n: [(std(X), y) for X, y in episodes(sources[n])] for n in names}
    model = SupLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for u in range(updates):
        pick = [eps[names[int(rng.integers(len(names)))]] for _ in range(batch)]
        pick = [e[int(rng.integers(len(e)))] for e in pick]
        T = min(len(x) for x, _ in pick)
        xb = torch.tensor(np.stack([x[:T] for x, _ in pick]))
        yb = torch.tensor(np.stack([y[:T] for _, y in pick]), dtype=torch.float32)
        opt.zero_grad()
        logits, _ = model(xb)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (u + 1) % 500 == 0:
            print(f"  SUP-LSTM seed {seed} update {u + 1}: loss {loss.item():.4f}", flush=True)
    return model, float(loss)


# ------------------------------------------- adapters for flag_episode
class LogRegPolicy:
    """Watchdog-style predict(): carries the episode's past observations as recurrent state."""

    def __init__(self, lin, std):
        self.lin, self.std = lin, std

    def predict(self, x, state=None, episode_start=None, deterministic=True):
        hist = [] if (state is None or episode_start[0]) else state
        xs = self.std(x[0])
        prev = np.mean(hist[-HISTORY:], axis=0) if hist else np.zeros_like(xs)
        with torch.no_grad():
            p = torch.sigmoid(self.lin(torch.tensor(np.concatenate([xs, prev])[None]))).item()
        return np.array([int(p > 0.5)]), hist + [xs]


class LSTMPolicy:
    def __init__(self, model, std):
        self.model, self.std = model, std

    def predict(self, x, state=None, episode_start=None, deterministic=True):
        if episode_start is not None and episode_start[0]:
            state = None
        with torch.no_grad():
            logit, state = self.model(torch.tensor(self.std(x[0]))[None, None], state)
        return np.array([int(logit.item() > 0.0)]), state


def label_rule_scored(src: dict, det) -> dict:
    """LABEL-RULE scored like score_source: step flags are the causal labels themselves."""
    flags_all = causal_labels(src)
    off = src["ep_offsets"]
    flags_by_ep = [flags_all[a:b] for a, b in zip(off[:-1], off[1:])]
    orders = orders_from_array(src["orders"])
    dec, delay = order_flags(orders, flags_by_ep)
    return {"step_flags": flags_all, "step_labels": src["y"], "order_truth": [o.manipulative for _, o in orders],
            "wd_orders": dec, "wd_delay": delay, "rule_orders": [det.flags(o) for _, o in orders],
            "rule_delay": [float(o.lifetime) if det.flags(o) else float("nan") for _, o in orders],
            "episodes_flagged": float("nan")}


# ------------------------------------------------------------ scoring
def load_policies(std: Standardiser) -> dict[str, list]:
    pols = {"LOGREG": [], "SUP-LSTM": []}
    for seed in SEEDS:
        lin = nn.Linear(2 * OBS_DIM, 1)
        lin.load_state_dict(torch.load(OUT / f"logreg_s{seed}.pt"))
        pols["LOGREG"].append(LogRegPolicy(lin.eval(), std))
        m = SupLSTM()
        m.load_state_dict(torch.load(OUT / f"suplstm_s{seed}.pt"))
        pols["SUP-LSTM"].append(LSTMPolicy(m.eval(), std))
    return pols


def summarise(scored: dict, attack_prefix: str | None) -> dict:
    """Order-level metrics over a dataset {name: scored}: attack recall, legitimate FPR, pooled F1."""
    allc = combine(list(scored.values()))
    legit = [v for n, v in scored.items() if n.startswith(("HONEST", "FLICKER", "ACCUM", "SAMESIDE", "REVERT"))]
    attack = [v for n, v in scored.items() if attack_prefix and n.startswith(attack_prefix)] or \
             [v for n, v in scored.items() if np.any(v["order_truth"])]
    return {"recall": combine(attack)["watchdog_orders"]["recall"] if attack else float("nan"),
            "legitimate FPR": combine(legit)["watchdog_orders"]["fpr"] if legit else float("nan"),
            "pooled F1": allc["watchdog_orders"]["f1"],
            "rule recall": combine(attack)["rule_orders"]["recall"] if attack else float("nan"),
            "rule legitimate FPR": combine(legit)["rule_orders"]["fpr"] if legit else float("nan"),
            "rule pooled F1": allc["rule_orders"]["f1"],
            "median delay (steps)": allc["watchdog_median_delay"]}


def evaluate(datasets: dict, std: Standardiser, det) -> dict:
    from sb3_contrib import RecurrentPPO
    detectors: dict[str, list] = {"LABEL-RULE": [None]}
    detectors.update(load_policies(std))
    detectors["RL WATCHDOG (cov)"] = []
    for tag in WATCHDOG_TAGS:
        path = ROOT / "checkpoints" / "WATCHDOG" / tag / "model.zip"
        detectors["RL WATCHDOG (cov)"].append((RecurrentPPO.load(path, device="cpu"), load_obs_norm(path.parent)))
    out = {}
    t0 = time.time()
    for ds_name, (data, prefix) in datasets.items():
        out[ds_name] = {}
        for det_name, models in detectors.items():
            per_seed = []
            for m in models:
                if det_name == "LABEL-RULE":
                    scored = {n: label_rule_scored(s, det) for n, s in data.items()}
                elif det_name.startswith("RL"):
                    scored = {n: score_source(m[0], s, det, m[1]) for n, s in data.items()}
                else:
                    scored = {n: score_source(m, s, det, None) for n, s in data.items()}
                per_seed.append(summarise(scored, prefix))
            out[ds_name][det_name] = {k: dict(zip(("mean", "ci95"), mean_ci95([p[k] for p in per_seed])))
                                      for k in per_seed[0]}
        print(f"[{time.time() - t0:5.0f}s] scored {ds_name}", flush=True)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "final"])
    args = p.parse_args()
    torch.set_num_threads(4)

    sources = train_sources()
    allX = np.vstack([np.clip(s["X"], -OBS_BOUND, OBS_BOUND) for s in sources.values()])
    std = Standardiser(allX.mean(0), allX.std(0) + 1e-6)
    det, _ = tune([o for n, s in load_split("train").items() for _, o in orders_from_array(s["orders"])])

    if args.mode == "train":
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(OUT / "standardiser.npz", mean=std.mean, std=std.std)
        for seed in SEEDS:
            lin, l1 = fit_logreg(sources, std, seed)
            torch.save(lin.state_dict(), OUT / f"logreg_s{seed}.pt")
            model, l2 = fit_lstm(sources, std, seed)
            torch.save(model.state_dict(), OUT / f"suplstm_s{seed}.pt")
            print(f"seed {seed}: logreg loss {l1:.4f}, sup-lstm loss {l2:.4f}", flush=True)
        datasets = {"original test split": (load_split("test"), None),
                    "VARIED-SPOOF + ACCUM test split": (load_split("test", EXT_DATA_DIR), "VARIED")}
        name = "baselines_dev"
    else:
        datasets = {"original test split": (load_split("test"), None),
                    "held-out split (SCRIPTED-ATK, AMZN)": (load_split("heldout"), "SCRIPTED"),
                    "LATEBURST-ATK": (load_split("lateburst"), "LATEBURST"),
                    "DRIP-ATK": (load_split("drip"), "DRIP"),
                    "REVERSAL-ATK": (load_split("reversal"), "REVERSAL")}
        name = "baselines_final"
    results = evaluate(datasets, std, det)
    payload = {"mode": args.mode, "seeds": SEEDS, "rule": {"n_mult": det.n_mult, "m_events": det.m_events},
               "results": results}
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (ROOT / "results" / f"{name}.md").write_text(render(payload), encoding="utf-8")
    print(f"wrote results/{name}.md")
    return 0


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def render(pl: dict) -> str:
    title = ("Baselines on the test splits (development)" if pl["mode"] == "train"
             else "Baselines on every held-out dataset (scored once, after the baselines were committed)")
    L = [f"# {title}", "",
         "Order level. Mean ± 95% Student-t CI over 3 training seeds (LABEL-RULE is deterministic). LABEL-RULE is the "
         "definition of the label written as a rule: it is the ceiling, perfect by construction, not a competitor. "
         f"Size + fast-cancel rule: N = {pl['rule']['n_mult']}, M = {pl['rule']['m_events']} events.", ""]
    for ds, block in pl["results"].items():
        L += [f"## {ds}", "", "| Detector | Recall | Legitimate FPR | Pooled F1 | Median delay (steps) |", "|---|---|---|---|---|"]
        for det_name, m in block.items():
            cell = lambda k: f"{_f(m[k]['mean'])} ± {_f(m[k]['ci95'])}" if np.isfinite(m[k]["ci95"]) else _f(m[k]["mean"])  # noqa: E731
            L.append(f"| {det_name} | {cell('recall')} | {cell('legitimate FPR')} | {cell('pooled F1')} | "
                     f"{_f(m['median delay (steps)']['mean'], 1)} |")
        r = next(iter(block.values()))
        L.append(f"| Size + fast-cancel rule | {_f(r['rule recall']['mean'])} | {_f(r['rule legitimate FPR']['mean'])} | "
                 f"{_f(r['rule pooled F1']['mean'])} | — |")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
