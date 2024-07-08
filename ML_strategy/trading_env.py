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

def flatten_dict(dct):
    return np.array([u for _,v in dct.items() for u in np.array(v).ravel()])

def unflatten_dictarr(arr, memory_length=30):
    return {
        "dotw" : arr[0],
        "close" : arr[1:memory_length+1],
        "vol" : arr[memory_length+1:2*memory_length+1],
        "ema" : arr[2*memory_length+1:2*memory_length+1+4],
        "cash" : arr[-2],
        "long" : arr[-1],
    }

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, env_config):#, render_mode=None, memory_length=30, episode_length=90):
        self.render_mode = None
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]
        self.memory_length = 30
        self.episode_length = 90
        # self.observation_space = gym.spaces.Dict({
        #     "dotw" : gym.spaces.Discrete(7), # Day of the week (Mon, Tue, Wed, Thu, Fri)
        #     "close" : gym.spaces.Box(low=0, high=np.inf, shape=(self.memory_length,)), # Close prices - last 30 days
        #     "vol" : gym.spaces.Box(low=0, high=np.inf, shape=(self.memory_length,)), # Trading volume - last 30 days
        #     "ema" : gym.spaces.Box(low=0, high=np.inf, shape=(4,)), # EMA60, EMA120, EMA240, EMA480
        #     "cash" : gym.spaces.Box(low=0, high=np.inf), # Cash position
        #     "long" : gym.spaces.Box(low=0, high=np.inf), # Number of stocks held
        # })
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(2*self.memory_length+7,))
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf) # Stock count to buy (+) or sell (-)

    def _get_obs(self):
        curr_index = self.startindex+self.day_index
        last_days = lambda length : self.stock_data.iloc[np.max([curr_index-length+1,0]):curr_index+1]
        if np.isnan(np.nanmean(last_days(480)["close"])): print(last_days(480)["close"])
        return_value = flatten_dict({
            "dotw" : self.stock_data.iloc[curr_index]["time"]//(60*60*24) % 7,
            "close" : last_days(self.memory_length)["close"],
            "vol" : last_days(self.memory_length)["Volume"],
            "ema" : np.array([np.nanmean(last_days(60)["close"]), np.nanmean(last_days(120)["close"]), np.nanmean(last_days(240)["close"]), np.nanmean(last_days(480)["close"])]),
            "cash" : self.cash,
            "long" : self.long_position,
        })
        return return_value

    def reset(self, seed=None, options={"starting_equity": 100}):
        # seed = 0
        super().reset(seed=seed)

        self.stock_data = pd.read_csv("../data/BATS_QQQ.csv")
        self.startindex = np.random.randint(0,self.stock_data.shape[0]-self.episode_length)

        self.day_index = self.memory_length-1
        self.long_position = 0
        self.cash = 100 # options["starting_equity"]
        return self._get_obs(), {}
    
    def step(self, action):
        action = action[0]
        # try:
        #     assert(isinstance(action, float))
        # except AssertionError:
        #     print("AssertionError. Action had type:", type(action))
        curr_index = self.startindex+self.day_index
        curr_price = self.stock_data.iloc[curr_index]["close"]
        action = np.clip(action, a_min=-self.long_position, a_max=self.cash/curr_price)
        
        self.long_position = np.clip(self.long_position + action, a_min=0, a_max=None)
        reward = -action * curr_price
        self.cash += np.clip(reward, a_min=0, a_max=None)
        
        terminated = (self.day_index == self.episode_length-1)
        if terminated:
            reward += self.cash + self.long_position*self.stock_data.iloc[curr_index]["close"]
        self.day_index += 1
        assert(isinstance(reward, float))
        return self._get_obs(), reward, terminated, False, {}
    
gym.envs.registration.register(
     id="TradingEnv-v1",
     entry_point="trading_env:TradingEnv",
)