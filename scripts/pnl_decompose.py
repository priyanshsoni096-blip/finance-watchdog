"""Where does a trained Spoofer's PnL come from?

Usage:
    python scripts/pnl_decompose.py SPOOFER-04 main [episodes] [--stochastic]

--stochastic samples actions from the trained policy instead of taking the most likely one.

Per trade, fill = historical touch + applied spoof shift + self-impact, so episode PnL splits into:
  spoof gain       profit from trading at a spoof-shifted price (the manipulation itself)
  spread           cost of crossing the spread
  self-impact      cost of the agent's own market-order volume
  liquidation      closing the position at episode end, relative to the mid
  inventory drift  everything else: mark-to-mid moves of held inventory (directional exposure)

A counterfactual run of the same policy with impact_lambda=0 (no spoof effect, no self-impact) shows how
much profit survives without manipulation. Headline PnL alone hid an env exploit twice in this project;
run this before trusting any profitable Spoofer.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from stable_baselines3 import PPO  # noqa: E402
from train_spoofer import AGENTS  # noqa: E402

from env.lob_env import BUY, EnvConfig, LimitOrderBookEnv  # noqa: E402
from env.lobster_data import load_day  # noqa: E402
from env.normalization import reference_stats  # noqa: E402


def decompose(agent: str, tag: str, episodes: int = 5, impact_lambda=None, deterministic: bool = True) -> dict:
    import torch
    torch.manual_seed(0)  # reproducible action sampling when deterministic=False
    spec = AGENTS[agent]
    folder = ROOT / "checkpoints" / agent / tag
    env_line = next(l for l in (folder / "config.txt").read_text().splitlines() if l.startswith("env="))
    saved = ast.literal_eval(env_line[4:])
    model = PPO.load(folder / "model.zip", device="cpu")
    d = load_day(spec["ticker"])
    s = reference_stats(d)
    # the checkpoint's saved granularity wins over the roster's, since that is what the model was trained on
    kwargs = {**spec["cfg"], "events_per_step": saved["events_per_step"], "episode_len": saved["episode_len"]}
    if impact_lambda is not None:
        kwargs["impact_lambda"] = impact_lambda
    env = LimitOrderBookEnv(EnvConfig(ticker=spec["ticker"], **kwargs), day=d, stats=s)
    tot = dict(pnl=0.0, spoof=0.0, spread=0.0, self_impact=0.0, liquidation=0.0, trades=0.0, max_inv_lots=0.0)
    for ep in range(episodes):
        obs, _ = env.reset(seed=123_000 + 17 * ep)
        while True:
            a = int(model.predict(obs, deterministic=deterministic)[0])
            t, v0 = env.t, env.volume
            obs, _, te, tr, info = env.step(a)
            if info["trade_price"] is not None:
                sgn = 1 if a == BUY else -1
                touch = d.ask_price[t, 0] if a == BUY else d.bid_price[t, 0]
                self_avg = env.self_shift(v0 + sgn * env.lot / 2, t)
                tot["spread"] -= env.lot * abs(touch - d.mid[t])
                tot["spoof"] -= sgn * env.lot * (info["trade_price"] - touch - self_avg)
                tot["self_impact"] -= sgn * env.lot * self_avg
                tot["trades"] += 1
            tot["max_inv_lots"] = max(tot["max_inv_lots"], abs(info["inventory"]) / env.lot)
            liq = info.get("liquidation") or {}
            if liq:
                tot["liquidation"] += liq["qty"] * (liq["price"] - d.mid[env.t])
            if te or tr:
                break
        tot["pnl"] += info["pnl"]
    for k in ("pnl", "spoof", "spread", "self_impact", "liquidation", "trades"):
        tot[k] /= episodes
    tot["drift"] = tot["pnl"] - tot["spoof"] - tot["spread"] - tot["self_impact"] - tot["liquidation"]
    tot.update(ticker=spec["ticker"], events_per_step=saved["events_per_step"])
    return tot


def main() -> int:
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    agent, tag = pos[0], pos[1]
    episodes = int(pos[2]) if len(pos) > 2 else 5
    deterministic = "--stochastic" not in flags
    mode = "argmax actions" if deterministic else "sampled actions"
    for label, lam in (("calibrated impact", None), ("impact_lambda=0 (counterfactual)", 0.0)):
        r = decompose(agent, tag, episodes, lam, deterministic)
        print(f"{agent}/{tag} {r['ticker']} events/step={r['events_per_step']} {mode} | {label:32s} per episode: "
              f"PnL {r['pnl']:+10.1f} = spoof gain {r['spoof']:+10.1f} + spread {r['spread']:+10.1f} "
              f"+ self-impact {r['self_impact']:+8.1f} + liquidation {r['liquidation']:+8.1f} "
              f"+ inventory drift {r['drift']:+10.1f} | trades {r['trades']:.0f} max |inv| {r['max_inv_lots']:.0f} lots",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
