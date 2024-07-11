from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

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
# def ema(seq, lifetime):
#     return pd.Series(seq).ewm(halflife=lifetime/2).mean().iloc[-1]

# === STRATEGIES ===

class Strategy(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def get_action(self, state):
        pass

class PPOStrategy(Strategy):
    def __init__(self, quiet=False):
        checkpoint_dir = "C:/Users/brand/AppData/Local/Temp/tmpzsk7vmkn"
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
        return state["cash"]/(state["close"][-1])
    
class DoNothingStrategy(Strategy):
    def get_action(self, state):
        return np.array([0])
    
# sell % when slope crosses from positive to negative, buy when vice versa
class RegressionSlopeStrategy(Strategy):
    def __init__(self, length, percentage):
        self.length = length
        self.percentage = percentage
        self.prev_slope = None

    def get_action(self, state):
        close = state["close"][-self.length:]
        indices = np.arange(self.length)
        slope = np.sum((close - np.mean(close)) * (indices - np.mean(indices))) / np.sum((indices - np.mean(indices))**2)

        action = np.array([0])
        if self.prev_slope is not None:
            # print(slope)
            if slope < 0 and self.prev_slope >= 0:
                action = -self.percentage * state["long"]
            elif slope > 0 and self.prev_slope <= 0:
                action = self.percentage * state["cash"] / (state["close"][-1])
        self.prev_slope = slope
        return action

# buy when MACD(30, 60) becomes positive, sell when becomes negative
class MACDStrategy(Strategy):
    def __init__(self, percentage):
        self.percentage = percentage
        self.prev_macd = None
    def get_action(self, state):
        macd = np.mean(state["close"][-30:]) - state["ema"][0]
        action = np.array([0])
        if self.prev_macd is not None:
            if macd > 0 and self.prev_macd <= 0:
                action = self.percentage * state["cash"] / (state["close"][-1])
            elif macd < 0 and self.prev_macd >= 0:
                action = -self.percentage * state["long"]
        self.prev_macd = macd
        return action

def _crossunder0(prev, curr):
    if prev is None: return False
    if prev < 0: return False
    if curr >= 0: return False
    return True
class JinStrategy(Strategy):
    def __init__(self,
                 rsi_threshold=70,
                 sell_percentage=0.01,
                 tdfi_thresh=-100):
            self.rsi_threshold = rsi_threshold
            self.sell_percentage = sell_percentage
            self.tdfi_thresh = tdfi_thresh

            self.close_minus_fast = None
            self.close_minus_slow = None
            self.close_minus_superslow = None
            self.last_buy_order_price = None

            self.remaining_buy_time = 0
            self.buy_value = 0

    def get_action(self, state):
        ema_fast = np.mean(state["close"][-20:]) # ema(state["close"], 20)
        ema_slow = state["ema"][0]
        ema_superslow = state["ema"][1]
        close = state["close"][-1]

        # calculate RSI
        percent_changes = (state["close"][-14:] - state["close"][-15:-1])/state["close"][-15:-1] * 100
        avg_percent_gain = np.mean(percent_changes[percent_changes > 0])
        avg_percent_loss = -np.mean(percent_changes[percent_changes < 0])
        rsi = 100 - (100/(1 + avg_percent_gain/avg_percent_loss))

        # calculate TDFI
        tdfi = (close - state["close"][-13])*state["vol"][-1]

        # sell whenever RSI is above the threshold AND close > EMA120
        if rsi > self.rsi_threshold and close > ema_slow:
            return -self.sell_percentage * state["long"]
        
        # while EMA_FAST > EMA_SLOW, buy aggressively
        if ema_fast > ema_slow:
            if _crossunder0(self.close_minus_fast, close - ema_fast):
                self.remaining_buy_time = 5
                self.buy_value = 0.02 * state["cash"]
            self.close_minus_fast = close - ema_fast
            if _crossunder0(self.close_minus_slow, close - ema_slow) and _crossunder0(self.close_minus_superslow, close - ema_superslow):
                self.remaining_buy_time = 5
                self.buy_value = 0.10 * state["cash"]
            self.close_minus_slow = close - ema_slow
            self.close_minus_superslow = close - ema_superslow
        else:
            if _crossunder0(self.close_minus_slow, close - ema_slow):
                self.remaining_buy_time = 5
                self.buy_value = 0.05 * state["cash"]
            if self.last_buy_order_price is not None and np.abs(close - self.last_buy_order_price)/self.last_buy_order_price > 0.10:
                self.remaining_buy_time = 5
                self.buy_value = 0.02 * state["cash"]
        
        # if the close > EMA_FAST, abort any buys
        if close > ema_fast:
            self.remaining_buy_time = 0
            self.buy_value = 0
        
        # resolve the buying
        if self.remaining_buy_time > 0:
            self.remaining_buy_time -= 1
            return self.buy_value / (state["close"][-1])
        
        return np.array([0])