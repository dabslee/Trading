import os
import numpy as np
from pull_data import download_stock_data
from trading_env import TradingEnv

def run_example_session(ticker="MSFT",
                        start_date_data="2020-01-01",
                        end_date_data="2023-12-31",
                        env_start_date="2021-01-01",
                        env_horizon_days=252, # Approx 1 trading year
                        initial_cash=10000,
                        cash_inflow=50,
                        render_mode='human', # 'human' or 'ansi'
                        data_folder="data"): # Relative to scripts
    """
    Runs a demonstration of the trading environment.
    1. Downloads stock data for the given ticker.
    2. Initializes the TradingEnv.
    3. Runs a loop taking some simple actions.
    4. Renders the environment.
    """
    print(f"Starting example session for ticker: {ticker}")

    # --- 1. Download Stock Data ---
    # pull_data.py and trading_env.py are in the same directory as main.py
    # The data_folder argument in download_stock_data is relative to pull_data.py's location.
    # If main.py, pull_data.py, trading_env.py are all in stock_trading_env/,
    # and data_folder is "data", then CSVs go into stock_trading_env/data/
    print(f"\nStep 1: Downloading data for {ticker} from {start_date_data} to {end_date_data}...")
    try:
        download_stock_data(
            tickers=[ticker],
            start_date=start_date_data,
            end_date=end_date_data,
            data_folder=data_folder
        )
        print(f"Data download successful for {ticker}.")
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        print("Please ensure yfinance is working and you have internet connectivity.")
        return

    # Verify that the data file was created as expected by the environment
    # trading_env.py expects data_folder to be relative to its own location.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    expected_data_file = os.path.join(script_dir, data_folder, f"{ticker.upper()}.csv")
    if not os.path.exists(expected_data_file):
        print(f"ERROR: Data file {expected_data_file} not found after download attempt.")
        print("There might be an issue with how `data_folder` is being interpreted by `download_stock_data` or `TradingEnv`.")
        return
    print(f"Data file confirmed at: {expected_data_file}")

    # --- 2. Initialize Trading Environment ---
    print(f"\nStep 2: Initializing Trading Environment for {ticker}...")
    try:
        env = TradingEnv(
            initial_cash=initial_cash,
            cash_inflow_per_step=cash_inflow,
            start_date_str=env_start_date,
            time_horizon_days=env_horizon_days,
            ticker=ticker,
            data_folder=data_folder, # This is also relative to trading_env.py
            render_lookback_window=40 # Days of history to show in render
        )
        print("Trading Environment initialized successfully.")
    except Exception as e:
        print(f"Error initializing TradingEnv for {ticker}: {e}")
        print("Ensure the data was downloaded correctly and covers the environment's start date, including MA calculation period.")
        return

    # --- 3. Run Simulation Loop ---
    print(f"\nStep 3: Running simulation for {env_horizon_days} steps...")
    obs, info = env.reset()
    env.render_mode = render_mode # Set render mode for the environment instance

    if render_mode == 'human':
        print("Human rendering mode enabled. A matplotlib window should appear.")
        print("If no window appears, your environment might not support GUI display.")
        print("Close the matplotlib window to continue after the simulation finishes.")


    total_reward = 0
    terminated = False
    truncated = False

    for i in range(env_horizon_days + 5): # Run a few steps beyond horizon to ensure termination
        # Simple example strategy: Buy if SMA5 > SMA20, Sell if SMA5 < SMA20
        # Observation: [current_cash, shares_held, current_price, sma5, sma20]
        current_price = obs[2]
        sma5 = obs[3]
        sma20 = obs[4]

        action_shares = 0.0
        if sma5 > sma20 and sma5 > 0 and sma20 > 0: # sma > 0 to avoid acting on initial NaN MAs if any slip through
            # Buy condition: try to buy 10 shares
            action_shares = 10.0
        elif sma5 < sma20 and sma5 > 0 and sma20 > 0:
            # Sell condition: try to sell 5 shares
            action_shares = -5.0

        action = np.array([action_shares], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if render_mode == 'ansi':
            env.render() # Explicit call for ANSI mode if needed (already in step for human)

        if i < env_horizon_days : # Only print regular step info within horizon
             print(f"Day {i+1}/{env_horizon_days} | Date: {info['current_date']} | Action: {action_shares:.1f} shares | "
                   f"Price: {info['current_price']:.2f} | Cash: {info['current_cash']:.2f} | "
                   f"Shares: {info['shares_held']:.1f} | Portfolio: {info['portfolio_value']:.2f} | Reward: {reward:.2f}")


        if terminated or truncated:
            print(f"\nEpisode finished after {i+1} steps.")
            print(f"Termination reason: {'Reached time horizon / end of data' if terminated else 'Truncated'}")
            print(f"Total reward accumulated: {total_reward:.2f}")
            print(f"Final portfolio state: Cash ${info['current_cash']:.2f}, Shares {info['shares_held']:.1f}, Value ${info['portfolio_value']:.2f}")
            break

    # --- 4. Close Environment ---
    print("\nStep 4: Closing environment.")
    env.close()
    print(f"\nExample session for {ticker} finished.")

if __name__ == "__main__":
    # Configuration for the example run
    TICKER_SYMBOL = "GOOG" # Try a different ticker
    DATA_START = "2019-01-01" # Longer history for MA calculation
    DATA_END = "2023-12-31"
    ENV_START = "2021-01-01" # Must be after DATA_START + MA window
    ENV_DAYS = 100           # Shorter horizon for quicker test
    RENDER_TYPE = 'human'    # Use 'ansi' if GUI is an issue or for faster runs

    # For agents, ensure data_folder pathing is correct based on CWD
    # Assuming main.py, pull_data.py, trading_env.py are in stock_trading_env/
    # And this script is run from stock_trading_env/
    # Then data_folder="data" correctly refers to stock_trading_env/data/
    # If script is run from /app/ (parent of stock_trading_env), then
    # python stock_trading_env/main.py would be used, and paths inside main.py
    # (like for data_folder) are relative to stock_trading_env/.

    # The current setup assumes main.py, pull_data.py, trading_env.py are siblings,
    # and 'data' is a sibling directory to them.

    run_example_session(
        ticker=TICKER_SYMBOL,
        start_date_data=DATA_START,
        end_date_data=DATA_END,
        env_start_date=ENV_START,
        env_horizon_days=ENV_DAYS,
        render_mode=RENDER_TYPE,
        data_folder="data" # Relative to the scripts' location
    )
