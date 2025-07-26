import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_env import TradingEnv
    from .policy_interface import PolicyCallable

def get_action(obs: np.ndarray, env: "TradingEnv") -> np.ndarray:
    """
    A policy based on SMA (Simple Moving Average) crossover.
    Buys if SMA5 > SMA20, Sells if SMA5 < SMA20.

    Args:
        obs: The current observation from the environment.
             Expected format: [current_cash, shares_held, current_price, sma5, sma20]
        env: The trading environment instance.

    Returns:
        A numpy array representing the action (number of shares to trade).
    """
    current_price_obs = obs[2] # Current price from observation
    sma5 = obs[3]
    sma20 = obs[4]

    action_shares = 0.0
    # Only trade if we have valid price and MA data
    if current_price_obs > 0 and sma5 > 0 and sma20 > 0:
        if sma5 > sma20:
            # Buy condition: try to buy 10 shares
            # This could be made more sophisticated, e.g., buy based on cash available
            # For now, sticking to the original logic in main.py
            action_shares = 10.0
        elif sma5 < sma20:
            # Sell condition: try to sell 5 shares
            # This could also be based on shares_held.
            # Sticking to original logic.
            action_shares = -5.0
            # Ensure we don't try to sell more shares than held (env handles this, but good practice for policy)
            # shares_held = obs[1]
            # action_shares = -min(5.0, shares_held) # Example: if policy should self-limit

    return np.array([action_shares], dtype=np.float32)

# For discoverability, assign to a variable satisfying the PolicyCallable type
policy: "PolicyCallable" = get_action
