from gymnasium.envs.registration import register

register(id="LimitOrderBook-v0", entry_point="env.lob_env:LimitOrderBookEnv")
