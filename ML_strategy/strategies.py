from abc import ABC, abstractmethod
import trading_env
import numpy as np

# === WARNING SUPPRESSION ===

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

# === HELPER METHODS ===

# returns True if seq1 crosses under seq2 at the index
def cross_under(seq1, seq2, index):
    if index < 1: return False
    seq1 = np.array(seq1)
    seq2 = np.array(seq2)
    return seq1[index] < seq2[index] and seq1[index-1] >= seq2[index-1]

# ema of a pandas dataframe
def ema(seq, lifetime, times):
    return seq.ewm(halflife=f"{lifetime/2} days", times=times).mean()

# === STRATEGIES ===

class Strategy(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def get_action(self, state):
        pass

class PPOStrategy(Strategy):
    def __init__(self, quiet=False):
        checkpoint_dir = "C:/Users/brand/AppData/Local/Temp/tmpeq55k3dl"
        if quiet:
            with suppress_stdout_stderr():
                from ray.rllib.algorithms.algorithm import Algorithm
                self.algo = Algorithm.from_checkpoint(checkpoint_dir)
        else:
            self.algo = Algorithm.from_checkpoint(checkpoint_dir)
    def get_action(self, state):
        return self.algo.compute_single_action(state)

class BuyAndHoldStrategy(Strategy):
    def get_action(self, state):
        state_dict = trading_env.unflatten_dictarr(state)
        return [state_dict["cash"]/(state_dict["close"][-1])]
    
class DoNothingStrategy(Strategy):
    def get_action(self, state):
        return [0]
    
# sell % when slope crosses from positive to negative, buy when vice versa
class RegressionSlopeStrategy(Strategy):

    def __init__(self, length, percentage):
        self.length = length
        self.percentage = percentage
        self.prev_slope = None

    def get_action(self, state):
        state_dict = trading_env.unflatten_dictarr(state)
        close = state_dict["close"][-self.length:]
        indices = np.arange(self.length)
        slope = np.sum((close - np.mean(close)) * (indices - np.mean(indices))) / np.sum((indices - np.mean(indices))**2)

        action = [0]
        if self.prev_slope is not None:
            # print(slope)
            if slope < 0 and self.prev_slope >= 0:
                action = [-self.percentage * state_dict["long"]]
            elif slope > 0 and self.prev_slope <= 0:
                action = [self.percentage * state_dict["cash"] / (state_dict["close"][-1])]
        self.prev_slope = slope
        return action

# class JinStrategy(Strategy):
#     def __init__(self,
#                  ema_fast=20, ema_slow=60, ema_superslow=120,
#                  rsi_threshold=70,
#                  slow_sell=0.01, quick_sell=0.05,
#                  tdfi_thresh=-100):
#             self.ema_fast = ema_fast
#             self.ema_slow = ema_slow
#             self.ema_superslow = ema_superslow
#             self.rsi_threshold = rsi_threshold
#             self.slow_sell = slow_sell
#             self.quick_sell = quick_sell
#             self.tdfi_thresh = tdfi_thresh
#     def get_action(self, state):
#         state_dict = trading_env.unflatten_dictarr(state)
        