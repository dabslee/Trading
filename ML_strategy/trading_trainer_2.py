import warnings
import ray
from ray.rllib.algorithms import ppo
from ray.tune.logger import pretty_print
import trading_env
import numpy as np
from matplotlib import pyplot as plt

def main():
    """
    Trains a trading agent using Deep Reinforcement Learning (PPO).
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    ray.shutdown()
    ray.init()

    # Configure the PPO algorithm
    config = {
        "env": trading_env.TradingEnv,
        "env_config": {
            "render_mode": None,
            "memory_length": 30,
            "episode_length": 90,
            "step_discount": trading_env.annual_to_daily_discount(0.98),
        },
        "framework": "torch",  # Or "tf" if you prefer TensorFlow
        "num_workers": 2,  # Number of rollout workers
        "num_gpus": 0,  # Number of GPUs to use (0 for CPU)
        # Add other PPO specific configurations as needed
        # For example:
        # "lr": 1e-4,
        # "gamma": 0.99,
        # "lambda": 0.95,
        # "kl_coeff": 0.2,
        # "sgd_minibatch_size": 128,
        # "num_sgd_iter": 10,
        # "rollout_fragment_length": 200,
        # "train_batch_size": 4000,
        # "model": {
        #     "fcnet_hiddens": [256, 256],
        #     "fcnet_activation": "tanh",
        # },
    }

    # Initialize the PPO agent
    # The old API `ppo.PPO(config=config)` is deprecated.
    # Use `config_obj = ppo.PPOConfig().environment(...).framework(...).rollouts(...).training(...).build()`
    algo = ppo.PPOConfig().environment(
        env=trading_env.TradingEnv,
        env_config=config["env_config"]
    ).framework(
        config["framework"]
    ).rollouts(
        num_rollout_workers=config["num_workers"]
    ).resources(
        num_gpus=config["num_gpus"]
    ).build()

    results_history = []
    num_iterations = 10  # Number of training iterations

    for i in range(num_iterations):
        print(f"Training iteration {i + 1}/{num_iterations}")
        result = algo.train()
        results_history.append(result)
        print(pretty_print(result))

        if (i + 1) % 10 == 0:  # Save checkpoint every 10 iterations
            checkpoint_dir = algo.save().checkpoint.path
            print(f"Checkpoint after iteration {i + 1} saved in directory: {checkpoint_dir}")

    # Save the final model
    final_checkpoint_dir = algo.save().checkpoint.path
    print(f"Final model saved in directory: {final_checkpoint_dir}")

    # Shutdown Ray
    ray.shutdown()

    # Plotting results (optional, can be commented out if not needed for a script)
    plot_training_results(results_history)

def plot_training_results(results_history):
    """
    Plots the average episode reward over training iterations.
    """
    if not results_history:
        print("No training results to plot.")
        return

    reward_mean = [res.get("episode_reward_mean", np.nan) for res in results_history]
    # Filter out potential NaNs if any iteration didn't produce this metric
    reward_mean_filtered = [r for r in reward_mean if not np.isnan(r)]
    iterations = np.arange(len(reward_mean_filtered))

    if not reward_mean_filtered:
        print("No valid episode_reward_mean found in results to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, reward_mean_filtered, marker='o', linestyle='-')
    plt.xlabel("Training Iteration")
    plt.ylabel("Mean Episode Reward")
    plt.title("Training Progress")
    plt.grid(True)

    # Try to save the plot to a file
    try:
        plot_filename = "trading_trainer_2_rewards.png"
        plt.savefig(plot_filename)
        print(f"Training results plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not save plot: {e}")

    # If running in a headless environment, plt.show() might not be desirable
    # or might cause issues. Consider if it's needed.
    # plt.show()


if __name__ == "__main__":
    main()
