"""Diagnostic: do the Watchdog's 46 surveillance features carry the manipulation signal?

Usage:
    python scripts/feature_signal_check.py [--features all|participant] [--no-heldout]

--features participant  uses only the 6 features of the monitored participant's own orders, not the 40 book
                        features, to test whether book features act as a stock-identity shortcut.
--no-heldout            skips heldout rows, for runs whose results inform a design decision.

Fits a class-balanced logistic regression on [current step features, mean of the previous 10 steps] using the
Watchdog `train` split and scores it step by step on `test` and `heldout` at a 0.5 threshold.

This is NOT a detector deliverable and does not answer the research question: it is supervised on the labels
directly, whereas the Watchdog only ever receives rewards. Its purpose is to separate two failure modes: if this
simple model separates positives but the Watchdog does not, the problem is RL optimisation, not missing
information in the observation.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.rollout import BOOK_DIM  # noqa: E402
from watchdog.dataset import load_split  # noqa: E402

HISTORY = 10


def windowed(src: dict, cols=slice(None)) -> np.ndarray:
    X, off = src["X"][:, cols], src["ep_offsets"]
    out = np.empty((len(X), X.shape[1] * 2), np.float32)
    for a, b in zip(off[:-1], off[1:]):
        ep = X[a:b]
        csum = np.cumsum(np.vstack([np.zeros((1, ep.shape[1]), np.float32), ep]), axis=0)
        idx = np.arange(len(ep))
        lo = np.maximum(idx - HISTORY, 0)
        prev_mean = (csum[idx] - csum[lo]) / np.maximum(idx - lo, 1)[:, None]
        out[a:b] = np.hstack([ep, prev_mean])
    return out


def stack(split: str, pred=lambda name: True, cols=slice(None)):
    data = load_split(split)
    names = [n for n in data if pred(n)]
    if not names:
        return None, None
    return np.vstack([windowed(data[n], cols) for n in names]), np.concatenate([data[n]["y"] for n in names])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--features", choices=["all", "participant"], default="all")
    p.add_argument("--no-heldout", action="store_true")
    args = p.parse_args()
    cols = slice(None) if args.features == "all" else slice(BOOK_DIM, None)
    print(f"features: {args.features}")

    torch.set_num_threads(2)
    torch.manual_seed(0)
    Xtr, ytr = stack("train", cols=cols)
    if Xtr is None:
        print("no training data; run watchdog/dataset.py first")
        return 1
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    X = torch.tensor((Xtr - mu) / sd)
    y = torch.tensor(ytr, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    pos_weight = torch.tensor((1 - ytr.mean()) / ytr.mean())
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y, pos_weight=pos_weight)
        loss.backward()
        opt.step()
    print(f"train loss {loss.item():.3f} (class-balanced)")

    buckets = [
        ("train (fit)", "train", lambda n: True),
        ("test: all sources", "test", lambda n: True),
        ("test: SPOOFER-02/03/04", "test", lambda n: n.startswith(("SPOOFER-02", "SPOOFER-03", "SPOOFER-04"))),
        ("test: legitimate only (HONEST+FLICKER)", "test", lambda n: n.startswith(("HONEST", "FLICKER"))),
        ("test: SPOOFER-01 (intent-less large orders)", "test", lambda n: n.startswith("SPOOFER-01")),
        ("heldout: SCRIPTED-ATK all stocks", "heldout", lambda n: n.startswith("SCRIPTED")),
        ("heldout: SPOOFER-05 + HONEST/FLICKER AMZN", "heldout", lambda n: not n.startswith("SCRIPTED")),
    ]
    # per source, so sparse spoofing (SPOOFER-03 GOOG, scripted attacker) is not hidden inside a pooled bucket
    buckets += [(f"test: {s}", "test", lambda n, s=s: n.startswith(s)) for s in ("SPOOFER-02", "SPOOFER-03", "SPOOFER-04")]
    buckets += [(f"heldout: SCRIPTED-ATK_{t}", "heldout", lambda n, t=t: n == f"SCRIPTED-ATK_{t}")
                for t in ("AAPL", "MSFT", "GOOG", "INTC", "AMZN")]
    for label, split, pred in buckets:
        if args.no_heldout and split == "heldout":
            continue
        Xs, ys = stack(split, pred, cols)
        if Xs is None:
            continue
        p = (torch.sigmoid(torch.tensor((Xs - mu) / sd) @ w + b) > 0.5).detach().numpy()
        tp, fp = np.sum(p & ys), np.sum(p & ~ys)
        fn, tn = np.sum(~p & ys), np.sum(~p & ~ys)
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        print(f"{label:44s} steps={len(ys):7d} positive={ys.mean():.3f} precision={precision:.2f} "
              f"recall={recall:.2f} fpr={fp / max(fp + tn, 1):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
