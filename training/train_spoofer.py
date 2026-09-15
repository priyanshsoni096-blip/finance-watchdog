"""Phase 2 — train one Spoofer from the population with Stable-Baselines3 PPO.

Usage:
    python training/train_spoofer.py SPOOFER-01 --timesteps 200000
    python training/train_spoofer.py SPOOFER-04 --timesteps 20000 --tag smoke

Writes checkpoints/<AGENT>/<tag>/model.zip plus progress.csv (one row per rollout) with the
behaviour statistics needed to show whether genuine spoof-and-cancel emerges (handoff §4.6 #2):
action frequencies, trades made while a spoof rested on the opposite side, cancels, run-overs.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from env.lob_env import (BUY, CANCEL, SELL, SPOOF_BUY, SPOOF_SELL, EnvConfig,  # noqa: E402
                         LimitOrderBookEnv)
from env.lobster_data import load_day  # noqa: E402
from env.normalization import reference_stats  # noqa: E402

# Locked roster (handoff §5.1). SPOOFER-05 is trained but never shown to the Watchdog.
AGENTS = {
    "SPOOFER-01": dict(ticker="AAPL", seed=101, cfg=dict(inv_penalty=0.01)),
    "SPOOFER-02": dict(ticker="MSFT", seed=202, cfg=dict(inv_penalty=0.0001)),
    "SPOOFER-03": dict(ticker="GOOG", seed=303, cfg=dict(bid_only=True)),
    "SPOOFER-04": dict(ticker="INTC", seed=404, cfg=dict()),
    "SPOOFER-05": dict(ticker="AMZN", seed=505, cfg=dict()),
}
ACTION_NAMES = ["noop", "buy", "sell", "spoof_buy", "spoof_sell", "cancel"]


class BehaviourLog(BaseCallback):
    """Aggregate per-rollout behaviour from env infos and actions into progress.csv."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.t0 = time.time()
        self._reset()
        self._fh = open(path, "w", newline="")
        self._w = None

    def _reset(self):
        self.actions = Counter()
        self.manip_trades = 0   # sell while spoof buy rests, or buy while spoof sell rests
        self.run_overs = 0
        self.ep_pnl, self.ep_rew = [], []

    def _on_training_start(self):
        self.cur_rew = np.zeros(self.training_env.num_envs)

    def _on_step(self) -> bool:
        actions = self.locals["actions"]
        infos = self.locals["infos"]
        self.cur_rew += self.locals["rewards"]
        for i, (a, info) in enumerate(zip(actions, infos)):
            a = int(a)
            self.actions[a] += 1
            self.run_overs += len(info.get("run_over", []))
            sides = info.get("pre_spoof_sides", ())
            if (a == SELL and 1 in sides) or (a == BUY and -1 in sides):
                self.manip_trades += 1
            if self.locals["dones"][i]:
                self.ep_pnl.append(info.get("final_pnl", info["pnl"]))
                self.ep_rew.append(self.cur_rew[i])
                self.cur_rew[i] = 0.0
        return True

    def _on_rollout_end(self):
        total = max(sum(self.actions.values()), 1)
        row = {
            "timesteps": self.num_timesteps,
            "wall_s": round(time.time() - self.t0, 1),
            "episodes": len(self.ep_rew),
            "ep_reward_mean": float(np.mean(self.ep_rew)) if self.ep_rew else np.nan,
            "ep_pnl_mean": float(np.mean(self.ep_pnl)) if self.ep_pnl else np.nan,
            **{f"frac_{n}": self.actions[i] / total for i, n in enumerate(ACTION_NAMES)},
            "manip_trades_per_1k": 1000 * self.manip_trades / total,
            "run_overs_per_1k": 1000 * self.run_overs / total,
        }
        if self._w is None:
            self._w = csv.DictWriter(self._fh, fieldnames=list(row))
            self._w.writeheader()
        self._w.writerow(row)
        self._fh.flush()
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        # cur_rew is NOT cleared here: episodes (2000 steps) span several rollouts
        self._reset()

    def _on_training_end(self):
        self._fh.close()


class SpoofSideInfo(LimitOrderBookEnv):
    """Adds which spoof sides were resting *before* the action, so a trade against a resting
    spoof can be counted, and the episode's final PnL before auto-reset."""

    def step(self, action):
        pre = tuple(s.side for s in self.spoofs)
        obs, r, term, trunc, info = super().step(action)
        info["pre_spoof_sides"] = pre
        if term or trunc:
            info["final_pnl"] = info["pnl"]
        return obs, r, term, trunc, info


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("agent", choices=sorted(AGENTS))
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--tag", default="main")
    p.add_argument("--seed", type=int, default=None, help="override roster seed (multi-seed runs)")
    # measured on 12k steps: 1 thread 39.3 s, 2 threads 34.7 s, 4 threads 35.3 s. 2 lets two
    # Spoofers train in parallel on this 4-core laptop without slowing each other.
    p.add_argument("--threads", type=int, default=2)
    # Exploration / credit-assignment knobs. On the v3 env every Spoofer converged to never trading,
    # although a scripted "spoof, wait ~50 events, trade, cancel" policy earns ~$4k/episode on
    # MSFT/INTC; defaults reproduce the original settings.
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--events-per-step", type=int, default=1)
    p.add_argument("--episode-len", type=int, default=None, help="agent steps; default keeps 2000 events")
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    spec = AGENTS[args.agent]
    seed = spec["seed"] if args.seed is None else args.seed
    episode_len = args.episode_len if args.episode_len is not None else 2000 // args.events_per_step
    cfg = EnvConfig(ticker=spec["ticker"], events_per_step=args.events_per_step, episode_len=episode_len,
                    **spec["cfg"])
    day = load_day(cfg.ticker)
    stats = reference_stats(day)

    def factory(rank):
        def _make():
            env = SpoofSideInfo(cfg, day=day, stats=stats)
            env.reset(seed=seed + rank)
            return env
        return _make

    vec = DummyVecEnv([factory(i) for i in range(args.n_envs)])
    out = ROOT / "checkpoints" / args.agent / args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.txt").write_text(f"agent={args.agent}\nseed={seed}\nenv={asdict(cfg)}\n"
                                    f"lambda={vec.envs[0].impact.lam}\nent_coef={args.ent_coef}\n"
                                    f"gamma={args.gamma}\n")

    n_steps = 4000 // args.n_envs  # train_batch_size = 4000 (handoff §5.3)
    model = PPO("MlpPolicy", vec, learning_rate=1e-4, n_steps=n_steps, batch_size=128, gamma=args.gamma,
                ent_coef=args.ent_coef,
                policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.ReLU),
                seed=seed, verbose=0, device="cpu")
    model.learn(total_timesteps=args.timesteps, callback=BehaviourLog(out / "progress.csv"))
    model.save(out / "model")
    print(f"saved {out / 'model.zip'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
