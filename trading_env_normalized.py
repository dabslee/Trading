import gymnasium as gym
import numpy as np
from trading_env import TradingEnv

class NormalizedTradingEnv(gym.Wrapper):
    """
    A wrapper for the TradingEnv that normalizes the observation and action spaces.
    """
    def __init__(self, env):
        super().__init__(env)

        # Original observation space for reference
        self.original_obs_space = env.observation_space

        # Define the normalized observation space (e.g., all values between 0 and 1)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=self.original_obs_space.shape, dtype=np.float32
        )

        # Action space remains the same, but we will scale the output of the agent
        # in the step function. For this example, let's assume the agent outputs
        # actions in the range [-1, 1] which we scale.
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)


    def _normalize_obs(self, obs):
        """Normalize the observation."""
        # This is a simple normalization. A more robust approach would use running means and stds.
        low = self.original_obs_space.low
        high = self.original_obs_space.high

        # Replace inf with a large number for normalization
        high[high == np.inf] = 1e6

        normalized_obs = (obs - low) / (high - low)
        return np.clip(normalized_obs, 0, 1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._normalize_obs(obs), info

    def step(self, action):
        # Scale the action from [-1, 1] to the original action space
        # For simplicity, let's scale it to a max of 10 shares per step
        scaled_action = action * 10

        obs, reward, terminated, truncated, info = self.env.step(scaled_action)

        # Also, let's try to improve the reward function slightly to encourage profit.
        # A simple reward shaping could be the change in portfolio value.
        new_reward = info['portfolio_value'] - self.last_portfolio_value
        self.last_portfolio_value = info['portfolio_value']

        return self._normalize_obs(obs), new_reward, terminated, truncated, info

    @property
    def render_mode(self):
        return self.env.render_mode

    @render_mode.setter
    def render_mode(self, mode):
        self.env.render_mode = mode

    def __getattr__(self, name):
        """Forward attribute calls to the underlying environment."""
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_portfolio_value = info['portfolio_value']
        return self._normalize_obs(obs), info
