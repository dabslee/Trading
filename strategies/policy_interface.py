from typing import Callable, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from trading_env import TradingEnv  # Avoid circular import at runtime

# A policy is a function that takes the current observation and environment instance,
# and returns an action (numpy array).
PolicyCallable = Callable[[np.ndarray, "TradingEnv"], np.ndarray]
