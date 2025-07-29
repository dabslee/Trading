import numpy as np
from stable_baselines3 import PPO
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_env import TradingEnv
    from .policy_interface import PolicyCallable

# --- Load the trained model ---
# Note: The model path is relative to the project root where the script is executed.
MODEL_PATH = "models/ppo_trading_model.zip"

# Load the PPO model only once when the module is imported.
# This avoids reloading the model on every single `get_action` call.
if os.path.exists(MODEL_PATH):
    try:
        model = PPO.load(MODEL_PATH)
        print(f"RL policy model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading RL model: {e}")
        model = None
else:
    print(f"Warning: RL model file not found at {MODEL_PATH}. The RL policy will not take any action.")
    model = None

def get_action(obs: np.ndarray, env: "TradingEnv") -> np.ndarray:
    """
    A policy that uses a pre-trained RL model to decide the action.

    Args:
        obs: The current observation from the environment.
        env: The trading environment instance.

    Returns:
        A numpy array representing the action (shares to trade).
    """
    if model:
        # The model's `predict` method returns the action and the next state (for recurrent policies)
        action, _states = model.predict(obs, deterministic=True)
        return action
    else:
        # Fallback action if the model failed to load: do nothing.
        return np.array([0.0], dtype=np.float32)

# For discoverability, assign to a variable satisfying the PolicyCallable type
policy: "PolicyCallable" = get_action
