import os
import numpy as np
import argparse # Added for policy selection
import importlib # Added for dynamic policy loading

from pull_data import download_stock_data
from trading_env import TradingEnv
# Import PolicyCallable for type hinting if needed, though not strictly necessary in main
# from strategies.policy_interface import PolicyCallable

def run_example_session(ticker="MSFT",
                        start_date_data="2020-01-01",
                        end_date_data="2023-12-31",
                        env_start_date="2021-01-01",
                        env_horizon_days=252, # Approx 1 trading year
                        initial_cash=10000,
                        cash_inflow=50,
                        render_mode='human', # 'human' or 'ansi'
                        data_folder="data", # Relative to scripts
                        policy_name="sma_crossover"):
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

    # --- 2b. Load Selected Policy ---
    print(f"\nStep 2b: Loading policy '{policy_name}'...")
    try:
        policy_module_name = f"strategies.{policy_name}_policy"
        policy_module = importlib.import_module(policy_module_name)
        selected_policy_get_action = policy_module.policy # Access the 'policy' variable which holds get_action
        print(f"Policy '{policy_name}' loaded successfully.")
    except ImportError:
        print(f"Error: Could not import policy '{policy_name}'. Make sure '{policy_module_name}.py' exists and is correct.")
        return
    except AttributeError:
        print(f"Error: Policy module '{policy_module_name}' does not have a 'policy' attribute or 'get_action' function.")
        return


    # --- 3. Run Simulation Loop ---
    print(f"\nStep 3: Running simulation for {env_horizon_days} steps using policy '{policy_name}'...")
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
        # Get action from the selected policy
        action = selected_policy_get_action(obs, env) # Pass obs and env
        action_shares = action[0] # Extract scalar for printing

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward # This total_reward is the sum of immediate rewards (which includes share value at the end)

        if render_mode == 'ansi' and not (terminated or truncated) : # Render if ansi and not yet done
             # Human mode renders inside env.step()
            env.render() 
        
        step_display_number = info.get("current_step", i+1) # Use actual step from info if available (it is self.current_step)

        print(f"Day {step_display_number} | Date: {info['current_date']} | Action: {action_shares:.1f} shares | "
              f"Price: {info['current_price']:.2f} | Cash: {info['current_cash']:.2f} | "
              f"Shares: {info['shares_held']:.1f} | Portfolio: {info['portfolio_value']:.2f} | "
              f"Step Reward: {reward:.2f}")

        if terminated or truncated:
            print(f"\n--- Episode Finished ---")
            print(f"Termination after {step_display_number} steps (Day {info['current_date']}).")
            print(f"Reason: {'Reached time horizon / end of data' if terminated else 'Truncated'}")
            
            # The last reward already includes the value of shares.
            # The total_reward is the sum of all rewards including this final one.
            # The final portfolio value is info['portfolio_value']
            # The total cash injected is initial_cash + (env.current_step * cash_inflow)
            
            total_cash_injected_final = initial_cash + (env.current_step * cash_inflow)
            
            print(f"\nFinal State:")
            print(f"  Cash: ${info['current_cash']:,.2f}")
            print(f"  Shares Held: {info['shares_held']:.2f} (Value: ${info['shares_held'] * info['current_price']:,.2f} at ${info['current_price']:.2f}/share)")
            print(f"  Final Portfolio Value (Cash + Shares): ${info['portfolio_value']:,.2f}")
            print(f"  Total Cash Injected During Episode: ${total_cash_injected_final:,.2f}")
            print(f"  Net Gain/Loss (Portfolio Value - Total Cash Injected): ${info['portfolio_value'] - total_cash_injected_final:,.2f}")
            print(f"  Sum of All Rewards (Total Reward): {total_reward:,.2f}")
            # Note: Total Reward should ideally reflect total change in value.
            # If initial portfolio value was initial_cash (0 shares), then
            # total_reward should be final_portfolio_value - total_cash_injected_through_inflows (excluding initial_cash here for this specific comparison)
            # Let's verify: sum of rewards = (final_cash + final_share_value) - initial_cash - sum(cash_inflows from trades)
            # The current total_reward is sum(trade_profit_or_loss) + sum(cash_inflow_per_step_rewards_if_any... no, cash_inflow is not part of reward) + final_share_value.
            # The reward is: cash_change_from_trade for normal steps. cash_change_from_trade + final_share_value for terminal.
            # So sum_rewards = sum(cash_changes_from_trades) + final_share_value_if_terminal.
            # This is not exactly net gain because it doesn't account for cash_inflows.
            # For an RL agent, this reward structure is fine. The printout helps a human understand.
            if render_mode == 'ansi' and (terminated or truncated): # One final render for ansi
                env.render()
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
    DEFAULT_POLICY = 'sma_crossover' # Default policy

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run a trading simulation.")
    parser.add_argument("--ticker", default=TICKER_SYMBOL, help="Stock ticker symbol (e.g., GOOG).")
    parser.add_argument("--data_start", default=DATA_START, help="Start date for historical data download (YYYY-MM-DD).")
    parser.add_argument("--data_end", default=DATA_END, help="End date for historical data download (YYYY-MM-DD).")
    parser.add_argument("--env_start", default=ENV_START, help="Start date for the simulation environment (YYYY-MM-DD).")
    parser.add_argument("--env_days", type=int, default=ENV_DAYS, help="Number of trading days for the simulation.")
    parser.add_argument("--render", default=RENDER_TYPE, choices=['human', 'ansi'], help="Render mode: 'human' or 'ansi'.")
    parser.add_argument("--policy", default=DEFAULT_POLICY, choices=['no_action', 'buy_and_hold', 'sma_crossover'],
                        help="Trading policy to use.")
    # Add other parameters like initial_cash, cash_inflow if needed as CLI args too.
    # For now, they use defaults from run_example_session.

    args = parser.parse_args()

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
        ticker=args.ticker,
        start_date_data=args.data_start,
        end_date_data=args.data_end,
        env_start_date=args.env_start,
        env_horizon_days=args.env_days,
        render_mode=args.render,
        policy_name=args.policy,
        data_folder="data" # Relative to the scripts' location (remains hardcoded for simplicity)
    )
