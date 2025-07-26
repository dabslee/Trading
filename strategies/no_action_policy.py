import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_env import TradingEnv
    from .policy_interface import PolicyCallable

def get_action(obs: np.ndarray, env: "TradingEnv") -> np.ndarray:
    """
    A policy that takes no action.

    Args:
        obs: The current observation from the environment.
        env: The trading environment instance.

    Returns:
        A numpy array representing the action (0 for hold).
    """
    return np.array(0)

# For discoverability, assign to a variable satisfying the PolicyCallable type
policy: "PolicyCallable" = get_action
