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
                        policy_name="sma_crossover",
                        verbose=True): # Control printing
    """
    Runs a demonstration of the trading environment.
    1. Downloads stock data for the given ticker.
    2. Initializes the TradingEnv.
    3. Runs a loop taking some simple actions.
    4. Renders the environment.
    5. Returns a dictionary of performance metrics.
    """
    if verbose:
        print(f"Starting example session for ticker: {ticker}")

    # --- 1. Download Stock Data ---
    if verbose:
        print(f"\nStep 1: Downloading data for {ticker} from {start_date_data} to {end_date_data}...")
    try:
        download_stock_data(
            tickers=[ticker],
            start_date=start_date_data,
            end_date=end_date_data,
            data_folder=data_folder
        )
        if verbose:
            print(f"Data download successful for {ticker}.")
    except Exception as e:
        if verbose:
            print(f"Error downloading data for {ticker}: {e}")
            print("Please ensure yfinance is working and you have internet connectivity.")
        return None

    # Verify that the data file was created as expected by the environment
    script_dir = os.path.dirname(os.path.abspath(__file__))
    expected_data_file = os.path.join(script_dir, data_folder, f"{ticker.upper()}.csv")
    if not os.path.exists(expected_data_file):
        if verbose:
            print(f"ERROR: Data file {expected_data_file} not found after download attempt.")
            print("There might be an issue with how `data_folder` is being interpreted by `download_stock_data` or `TradingEnv`.")
        return None
    if verbose:
        print(f"Data file confirmed at: {expected_data_file}")

    # --- 2. Initialize Trading Environment ---
    if verbose:
        print(f"\nStep 2: Initializing Trading Environment for {ticker}...")
    try:
        env = TradingEnv(
            initial_cash=initial_cash,
            cash_inflow_per_step=cash_inflow,
            start_date_str=env_start_date,
            time_horizon_days=env_horizon_days,
            ticker=ticker,
            data_folder=data_folder,
            render_lookback_window=40
        )
        if verbose:
            print("Trading Environment initialized successfully.")
    except Exception as e:
        if verbose:
            print(f"Error initializing TradingEnv for {ticker}: {e}")
            print("Ensure the data was downloaded correctly and covers the environment's start date, including MA calculation period.")
        return None

    # --- 2b. Load Selected Policy ---
    if verbose:
        print(f"\nStep 2b: Loading policy '{policy_name}'...")
    try:
        policy_module_name = f"strategies.{policy_name}_policy"
        policy_module = importlib.import_module(policy_module_name)
        selected_policy_get_action = policy_module.policy
        if verbose:
            print(f"Policy '{policy_name}' loaded successfully.")
    except ImportError:
        if verbose:
            print(f"Error: Could not import policy '{policy_name}'. Make sure '{policy_module_name}.py' exists and is correct.")
        return None
    except AttributeError:
        if verbose:
            print(f"Error: Policy module '{policy_module_name}' does not have a 'policy' attribute or 'get_action' function.")
        return None


    # --- 3. Run Simulation Loop ---
    if verbose:
        print(f"\nStep 3: Running simulation for {env_horizon_days} steps using policy '{policy_name}'...")
    obs, info = env.reset()
    env.render_mode = render_mode
    
    if render_mode == 'human' and verbose:
        print("Human rendering mode enabled. A matplotlib window should appear.")
        print("If no window appears, your environment might not support GUI display.")
        print("Close the matplotlib window to continue after the simulation finishes.")

    portfolio_value_history = []
    total_reward = 0
    terminated = False
    truncated = False

    for i in range(env_horizon_days + 5):
        portfolio_value_history.append(info.get('portfolio_value', initial_cash))

        action = selected_policy_get_action(obs, env)
        action_shares = action[0]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if render_mode == 'ansi' and not (terminated or truncated) and verbose:
            env.render() 
        
        if verbose:
            step_display_number = info.get("current_step", i+1)
            print(f"Day {step_display_number} | Date: {info['current_date']} | Action: {action_shares:.1f} shares | "
                  f"Price: {info['current_price']:.2f} | Cash: {info['current_cash']:.2f} | "
                  f"Shares: {info['shares_held']:.1f} | Portfolio: {info['portfolio_value']:.2f} | "
                  f"Step Reward: {reward:.2f}")

        if terminated or truncated:
            if verbose:
                step_display_number = info.get("current_step", i+1)
                print(f"\n--- Episode Finished ---")
                print(f"Termination after {step_display_number} steps (Day {info['current_date']}).")
                print(f"Reason: {'Reached time horizon / end of data' if terminated else 'Truncated'}")
            
            total_cash_injected_final = initial_cash + (env.current_step * cash_inflow)
            final_portfolio_value = info['portfolio_value']
            net_gain_loss = final_portfolio_value - total_cash_injected_final

            if verbose:
                print(f"\nFinal State:")
                print(f"  Cash: ${info['current_cash']:,.2f}")
                print(f"  Shares Held: {info['shares_held']:.2f} (Value: ${info['shares_held'] * info['current_price']:,.2f} at ${info['current_price']:.2f}/share)")
                print(f"  Final Portfolio Value (Cash + Shares): ${final_portfolio_value:,.2f}")
                print(f"  Total Cash Injected During Episode: ${total_cash_injected_final:,.2f}")
                print(f"  Net Gain/Loss (Portfolio Value - Total Cash Injected): ${net_gain_loss:,.2f}")
                print(f"  Sum of All Rewards (Total Reward): {total_reward:,.2f}")
                if render_mode == 'ansi' and (terminated or truncated):
                    env.render()
            
            # --- 4. Close Environment and Return Results ---
            env.close()

            # Ensure final portfolio value is recorded
            portfolio_value_history.append(final_portfolio_value)

            return {
                "final_portfolio_value": final_portfolio_value,
                "total_cash_injected": total_cash_injected_final,
                "net_gain_loss": net_gain_loss,
                "total_reward": total_reward,
                "portfolio_value_history": portfolio_value_history,
                "total_trades": len(env.trade_history)
            }
    
    # This part should ideally not be reached if horizon is met, but as a fallback:
    env.close()
    if verbose:
        print("\nStep 4: Closing environment.")
        print(f"\nExample session for {ticker} finished (ran out of loop steps).")
    return None # Indicate incomplete run

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
    
    results = run_example_session(
        ticker=args.ticker,
        start_date_data=args.data_start,
        end_date_data=args.data_end,
        env_start_date=args.env_start,
        env_horizon_days=args.env_days,
        render_mode=args.render,
        policy_name=args.policy,
        data_folder="data", # Relative to the scripts' location (remains hardcoded for simplicity)
        verbose=True # Always print details when run as a script
    )

    if results:
        print("\n--- Simulation Complete ---")
        print(f"Final portfolio value: ${results['final_portfolio_value']:,.2f}")
        print(f"Net gain/loss: ${results['net_gain_loss']:,.2f}")
        print(f"Total trades: {results['total_trades']}")
