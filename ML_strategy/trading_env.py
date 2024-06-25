import numpy as np
import gymnasium as gym
import pandas as pd

'''
We create a TradingEnv where the agent must decide at the
close of each day whether to buy or sell stocks, and if
so, how many. The environment selects a random sample of
90 days of data for a given episode.

We seek to use this to create a simple, time-independent
trading strategy. The state features observed by the
agent at any given time step are:
- Day of the week
- Close prices of the last 30 days
- Trade volume of the last 30 days
- MA60, MA120, MA240, MA480
- Current cash position, current long position
'''

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, memory_length=30, episode_length=90):
        self.memory_length = 30
        self.episode_length = 30
        self.observation_space = gym.spaces.Dict({
            "dotw" : gym.spaces.Discrete(5), # Day of the week (Mon, Tue, Wed, Thu, Fri)
            "close" : gym.spaces.Box(low=0, high=np.inf, shape=(memory_length,)), # Close prices - last 30 days
            "vol" : gym.spaces.Box(low=0, high=np.inf, shape=(1,)), # Trading volume - last 30 days
            "ema" : gym.spaces.Box(low=0, high=np.inf, shape=(4,)), # EMA60, EMA120, EMA240, EMA480
            "cash" : gym.spaces.Box(low=0, high=np.inf, shape=(1,)), # Cash position
            "long" : gym.spaces.Box(low=0, high=np.inf, shape=(2,)), # Number of stocks held
        })
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)) # Stock count to buy (+) or sell (-)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        curr_index = self.startindex+self.day_index
        last_days = lambda length : self.stock_data.iloc[np.max(curr_index-length+1,0):curr_index+1]
        return {
            "dotw" : self.stock_data.iloc[curr_index]["time"]//(60*60*24) % 7,
            "close" : last_days(self.memory_length)["close"],
            "vol" : last_days(self.memory_length)["Volume"],
            "ema" : [np.mean(last_days(60)["close"]), np.mean(last_days(120)["close"]), np.mean(last_days(240)["close"]), np.mean(last_days(480)["close"])],
            "cash" : self.cash,
            "long" : self.long_position,
        }

    def reset(self, seed=None, options={"starting_equity": 100}):
        super().reset(seed=seed)

        self.stock_data = pd.read_csv("data/BATS_TQQQ.csv")
        self.startindex = np.random.randint(0,self.stock_data.shape[0]-self.episode_length)

        self.day_index = self.memory_length-1
        self.long_position = 0
        self.cash = options["starting_equity"]
        return self._get_obs(), {}
    
    def step(self, action):
        curr_index = self.startindex+self.day_index
        curr_price = self.stock_data.iloc[curr_index]["close"]
        action = np.clip(action, amin=-self.long_position, amax=self.cash/curr_price)
        
        self.long_position += action
        reward = action * curr_price
        self.cash += reward
        
        terminated = (self.day_index == self.episode_length-1)
        if terminated:
            reward += self.cash + self.long_position*self.training_data.iloc[self.day_number]["close"]
        self.day_index += 1
        return self._get_obs(), reward, terminated, False, {}
    
gym.envs.registration.register(
     id="TradingEnv-v1",
     entry_point="trading_env:TradingEnv",
)