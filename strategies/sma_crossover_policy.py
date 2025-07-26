import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_env import TradingEnv
    from .policy_interface import PolicyCallable

def get_action(obs: np.ndarray, env: "TradingEnv") -> np.ndarray:
    """A policy based on SMA (Simple Moving Average) crossover.

    Buys if SMA5 > SMA20, Sells if SMA5 < SMA20.
    """
    sma5 = obs[3]
    sma20 = obs[4]

    if sma5 > sma20:
        # Buy action
        return np.array(1)
    elif sma5 < sma20:
        # Sell action
        return np.array(2)

    # Hold action
    return np.array(0)

# For discoverability, assign to a variable satisfying the PolicyCallable type
policy: "PolicyCallable" = get_action
