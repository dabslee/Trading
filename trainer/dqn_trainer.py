import os
import argparse
from stable_baselines3 import DQN
from trading_env import TradingEnv

def train_dqn(ticker, start_date, end_date, env_start_date, env_horizon_days,
              initial_cash, cash_inflow, model_save_path, data_folder="data",
              total_timesteps=20000):
    """
    Trains a DQN model on the trading environment.
    """
    print("--- Starting DQN Training ---")

    # --- 1. Create Environment ---
    env = TradingEnv(
        initial_cash=initial_cash,
        cash_inflow_per_step=cash_inflow,
        start_date_str=env_start_date,
        time_horizon_days=env_horizon_days,
        ticker=ticker,
        data_folder=data_folder
    )

    # --- 2. Create DQN Model ---
    # The model will be learning a policy for the environment.
    # We use the MultiInputPolicy because our observation space is a dictionary.
    # MlpPolicy is for flat observation spaces.
    model = DQN("MlpPolicy", env, verbose=1,
                learning_rate=0.0005,
                buffer_size=50000,
                learning_starts=1000,
                batch_size=32,
                tau=1.0,
                gamma=0.99,
                train_freq=(4, "step"),
                gradient_steps=-1,
                target_update_interval=1000,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                tensorboard_log="./dqn_trading_tensorboard/")

    # --- 3. Train the Model ---
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, log_interval=4)
    print("--- Training Finished ---")

    # --- 4. Save the Model ---
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN model for trading.")
    parser.add_argument("--ticker", default="GOOG", help="Stock ticker symbol.")
    parser.add_argument("--data_start", default="2019-01-01", help="Start date for data download.")
    parser.add_argument("--data_end", default="2023-12-31", help="End date for data download.")
    parser.add_argument("--env_start", default="2021-01-01", help="Start date for the simulation.")
    parser.add_argument("--env_days", type=int, default=252, help="Number of trading days for the simulation.")
    parser.add_argument("--initial_cash", type=int, default=10000, help="Initial cash.")
    parser.add_argument("--cash_inflow", type=int, default=100, help="Cash inflow per step.")
    parser.add_argument("--model_save_path", default="models/dqn_goog.zip", help="Path to save the trained model.")
    parser.add_argument("--timesteps", type=int, default=20000, help="Total timesteps for training.")
    args = parser.parse_args()

    train_dqn(
        ticker=args.ticker,
        start_date=args.data_start,
        end_date=args.data_end,
        env_start_date=args.env_start,
        env_horizon_days=args.env_days,
        initial_cash=args.initial_cash,
        cash_inflow=args.cash_inflow,
        model_save_path=args.model_save_path,
        total_timesteps=args.timesteps
    )
