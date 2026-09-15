"""Every source the Watchdog observes must share one decision granularity (events per agent step);
mixing granularities would silently change what one Watchdog step means across sources."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

from train_spoofer import AGENTS, DECISION_ENV  # noqa: E402

from env.lob_env import EnvConfig  # noqa: E402
from watchdog.dataset import sources  # noqa: E402


def test_decision_env_keeps_2000_event_episodes():
    assert DECISION_ENV["events_per_step"] * DECISION_ENV["episode_len"] == 2000


def test_every_spoofer_uses_decision_env():
    for name, spec in AGENTS.items():
        for key, value in DECISION_ENV.items():
            assert spec["cfg"].get(key) == value, f"{name} {key}"
        EnvConfig(ticker=spec["ticker"], **spec["cfg"])


def test_scripted_controls_use_decision_env():
    by_split = sources(tag="__no_such_checkpoint__")   # Spoofers skipped; controls only
    controls = [s for split in by_split.values() for s in split]
    assert controls
    for src in controls:
        assert src.name in {"HONEST", "FLICKER", "SCRIPTED-ATK"}
        for key, value in DECISION_ENV.items():
            assert src.cfg.get(key) == value, f"{src.name}_{src.ticker} {key}"
