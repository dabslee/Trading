import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_env import TradingEnv
    from .policy_interface import PolicyCallable

def get_action(obs: np.ndarray, env: "TradingEnv") -> np.ndarray:
    """
    A buy-and-hold policy. It buys shares with a percentage of available cash
    if cash is available and the price is positive. It never sells.

    Args:
        obs: The current observation from the environment.
             Expected format: [current_cash, shares_held, current_price, sma5, sma20]
        env: The trading environment instance.

    Returns:
        A numpy array representing the action (number of shares to buy, or 0.0).
    """
    current_cash = obs[0]
    current_price = obs[2]

    action_shares = 0.0

    # Buy with 95% of cash if price is valid
    if current_cash > 0 and current_price > 0:
        cash_to_use = current_cash * 0.95
        shares_to_buy = cash_to_use / current_price
        action_shares = shares_to_buy

    return np.array([action_shares], dtype=np.float32)

# For discoverability, assign to a variable satisfying the PolicyCallable type
policy: "PolicyCallable" = get_action
