from abc import ABC, abstractmethod
import trading_env

# Warning suppression
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

# Strategies
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