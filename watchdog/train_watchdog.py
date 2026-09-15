"""Train the WATCHDOG (RecurrentPPO, LSTM policy) on recorded trajectories of the frozen agent pool.

Usage:
    python watchdog/dataset.py --episodes 150             # build data first
    python watchdog/train_watchdog.py --timesteps 400000

Uses only the `train` split (never SPOOFER-05, SCRIPTED-ATK or any AMZN source).
Writes checkpoints/WATCHDOG/<tag>/model.zip and progress.csv with per-rollout step-level
precision / recall / false-positive rate on the training data.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sb3_contrib import RecurrentPPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from watchdog.dataset import DATA_DIR, load_split  # noqa: E402
from watchdog.watchdog_env import WatchdogEnv  # noqa: E402


class DetectionLog(BaseCallback):
    def __init__(self, path: Path):
        super().__init__()
        self.path, self.t0 = path, time.time()
        self._fh = open(path, "w", newline="")
        self._w = None
        self.tp = self.fp = self.fn = self.tn = 0
        self.reward = 0.0

    def _on_step(self) -> bool:
        self.reward += float(np.sum(self.locals["rewards"]))
        for info in self.locals["infos"]:
            f, y = info["flag"], info["label"]
            self.tp += f and y
            self.fp += f and not y
            self.fn += (not f) and y
            self.tn += (not f) and not y
        return True

    def _on_rollout_end(self):
        n = self.tp + self.fp + self.fn + self.tn
        row = {
            "timesteps": self.num_timesteps, "wall_s": round(time.time() - self.t0, 1),
            "reward_per_step": self.reward / max(n, 1),
            "positive_rate": (self.tp + self.fn) / max(n, 1),
            "flag_rate": (self.tp + self.fp) / max(n, 1),
            "precision": self.tp / (self.tp + self.fp) if self.tp + self.fp else float("nan"),
            "recall": self.tp / (self.tp + self.fn) if self.tp + self.fn else float("nan"),
            "fpr": self.fp / (self.fp + self.tn) if self.fp + self.tn else float("nan"),
        }
        if self._w is None:
            self._w = csv.DictWriter(self._fh, fieldnames=list(row))
            self._w.writeheader()
        self._w.writerow(row)
        self._fh.flush()
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        self.tp = self.fp = self.fn = self.tn = 0
        self.reward = 0.0

    def _on_training_end(self):
        self._fh.close()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=400_000)
    p.add_argument("--tag", default="main")
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--fp-cost", type=float, default=2.0)
    p.add_argument("--fn-cost", type=float, default=1.0)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    sources = load_split("train", Path(args.data))
    if not sources:
        print(f"no training data in {Path(args.data) / 'train'}; run watchdog/dataset.py first")
        return 1
    for name, s in sources.items():
        print(f"train source {name:18s} steps={len(s['y']):7d} positive={s['y'].mean():.3f}")

    def factory(rank):
        def _make():
            env = WatchdogEnv(sources, fn_cost=args.fn_cost, fp_cost=args.fp_cost)
            env.reset(seed=args.seed + rank)
            return env
        return _make

    vec = DummyVecEnv([factory(i) for i in range(args.n_envs)])
    out = ROOT / "checkpoints" / "WATCHDOG" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.txt").write_text(f"{vars(args)}\nsources={sorted(sources)}\n")

    model = RecurrentPPO("MlpLstmPolicy", vec, learning_rate=3e-4, n_steps=512, batch_size=512, n_epochs=5,
                         gamma=0.9, policy_kwargs=dict(lstm_hidden_size=64, net_arch=[128]),
                         seed=args.seed, verbose=0, device="cpu")
    model.learn(total_timesteps=args.timesteps, callback=DetectionLog(out / "progress.csv"))
    model.save(out / "model")
    print(f"saved {out / 'model.zip'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
