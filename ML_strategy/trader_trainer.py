import ray
from ray.rllib.algorithms import ppo
from ray.tune.logger import pretty_print
import trading_env

ray.shutdown()
ray.init()

algo = ppo.PPO(env=trading_env.TradingEnv, config={"env_config": {
    "render_mode" : None,
    "memory_length" : 30,
    "episode_length" : 90,
    "step_discount" : trading_env.annual_to_daily_discount(0.98),
}})

for i in range(10):
    print(f"Training episode {i}")
    result = algo.train()
    print(pretty_print(result))

checkpoint_dir = algo.save().checkpoint.path
print(f"Checkpoint saved in directory {checkpoint_dir}")