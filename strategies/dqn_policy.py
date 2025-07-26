import numpy as np
from stable_baselines3 import DQN
from strategies.policy_interface import PolicyCallable

# Global container for the loaded model to avoid reloading it on every call.
_model = None

def load_model(model_path="models/dqn_goog.zip"):
    """Loads the DQN model from the specified path."""
    global _model
    if _model is None:
        try:
            _model = DQN.load(model_path)
        except Exception as e:
            print(f"Error loading DQN model: {e}")
            # Depending on requirements, you might want to handle this more gracefully.
            # For now, we'll let it raise an exception if the model can't be loaded.
            raise
    return _model

def dqn_policy(obs: np.ndarray, env) -> np.ndarray:
    """
    A trading policy that uses a pre-trained DQN model to make decisions.

    Args:
        obs: The current observation from the environment.
        env: The environment instance, providing context if needed.

    Returns:
        An action (numpy array) to be taken in the environment.
    """
    # Path to the model can be configured here or passed differently.
    model_path = "models/dqn_goog.zip"
    model = load_model(model_path)

    # The model's `predict` method returns the action and the next state (if available).
    # We only need the action for the policy.
    # The `deterministic=True` argument means we are not using exploration noise.
    action, _states = model.predict(obs, deterministic=True)

    return action

# This is the function that will be imported by other scripts.
policy: PolicyCallable = dqn_policy
