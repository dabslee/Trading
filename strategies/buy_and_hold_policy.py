import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_env import TradingEnv
    from .policy_interface import PolicyCallable

def get_action(obs: np.ndarray, env: "TradingEnv") -> np.ndarray:
    """A buy-and-hold policy.

    It buys on the first step and then holds.
    """
    if env.current_step == 0:
        # Buy action
        return np.array(1)

    # Hold action
    return np.array(0)

# For discoverability, assign to a variable satisfying the PolicyCallable type
policy: "PolicyCallable" = get_action
