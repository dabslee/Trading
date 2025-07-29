import os
import argparse
# Note: The following warnings from TensorFlow can be safely ignored since we are using PyTorch.
# 2025-07-29 13:05:19.556621: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on.
# You may see slightly different numerical results due to floating-point round-off errors from different
# computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from trading_env_normalized import NormalizedTradingEnv
from trading_env import TradingEnv
from pull_data import download_stock_data

def train_rl_model(ticker, data_start, data_end, train_timesteps, model_path, initial_cash, cash_inflow, time_horizon):
    """
    Train a PPO model on the TradingEnv.
    """
    print("--- Starting RL Model Training ---")

    # --- 1. Download Data ---
    print(f"Downloading data for {ticker} from {data_start} to {data_end}...")
    download_stock_data(
        tickers=[ticker],
        start_date=data_start,
        end_date=data_end,
        data_folder="data"
    )

    # --- 2. Create Vectorized Environment ---
    print("Creating vectorized trading environment...")

    def make_env():
        env = TradingEnv(
            initial_cash=initial_cash,
            cash_inflow_per_step=cash_inflow,
            time_horizon_days=time_horizon,
            ticker=ticker,
            start_date_str=None,  # Use random start dates
        )
        return NormalizedTradingEnv(env)

    # Create a vectorized environment
    vec_env = make_vec_env(make_env, n_envs=4)

    # --- 3. Train the PPO Model ---
    print(f"Training PPO model for {train_timesteps} timesteps...")

    # Instantiate the PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="./ppo_trading_tensorboard/"
    )

    # Train the model
    model.learn(total_timesteps=train_timesteps)

    # --- 4. Save the Trained Model ---
    print(f"Saving trained model to {model_path}...")
    model.save(model_path)
    print(f"Model saved successfully.")

    # --- 5. Close Environment ---
    vec_env.close()
    print("--- Training Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL trading model.")
    parser.add_argument("--ticker", default="GOOG", help="Stock ticker symbol.")
    parser.add_argument("--data_start", default="2019-01-01", help="Start date for data download.")
    parser.add_argument("--data_end", default="2023-12-31", help="End date for data download.")
    parser.add_argument("--timesteps", type=int, default=25000, help="Total timesteps for training.")
    parser.add_argument("--model_path", default="models/ppo_trading_model.zip", help="Path to save the trained model.")
    parser.add_argument("--initial_cash", type=int, default=10000, help="Initial cash in the environment.")
    parser.add_argument("--cash_inflow", type=int, default=100, help="Cash inflow per step in the environment.")
    parser.add_argument("--time_horizon", type=int, default=252, help="Time horizon for each episode (days).")

    args = parser.parse_args()

    # Ensure the directory for the model exists
    model_dir = os.path.dirname(args.model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    train_rl_model(
        ticker=args.ticker,
        data_start=args.data_start,
        data_end=args.data_end,
        train_timesteps=args.timesteps,
        model_path=args.model_path,
        initial_cash=args.initial_cash,
        cash_inflow=args.cash_inflow,
        time_horizon=args.time_horizon
    )
