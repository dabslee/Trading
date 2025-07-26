import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 1}

    def __init__(self, initial_cash, cash_inflow_per_step, start_date_str, time_horizon_days, ticker,
                 data_folder="data", window_size=20, render_lookback_window=60):
        super().__init__()

        self.initial_cash = initial_cash
        self.cash_inflow_per_step = cash_inflow_per_step
        self.start_date_str = start_date_str
        self.time_horizon_days = time_horizon_days
        self.ticker = ticker.upper()
        self.window_size = window_size # For moving averages
        self.render_lookback_window = render_lookback_window # For rendering chart

        # Load data
        # Assuming data_folder is relative to this script's location (e.g., "data")
        # and this script is in stock_trading_env/
        # So, stock_trading_env/data/TICKER.csv
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_file_path = os.path.join(script_dir, data_folder, f"{self.ticker}.csv")
        
        if not os.path.exists(self.data_file_path):
            raise FileNotFoundError(f"Data file not found: {self.data_file_path}. "
                                    f"Run pull_data.py for ticker {self.ticker} first.")
        
        # yfinance, even for single tickers, sometimes saves with multi-level column headers
        # e.g., ('Close', 'AAPL'). The Date index is the first column.
        # Headers are on row 0 and 1. Index is column 0.
        try:
            self.df = pd.read_csv(self.data_file_path, header=[0, 1], index_col=0, parse_dates=True)
        except ValueError as e:
            # Fallback for potentially simpler CSVs (e.g. if pull_data changes or for user-provided CSVs)
            # This might happen if the CSV doesn't have multi-index columns as expected.
            print(f"Failed to read CSV with multi-index header, attempting simple read: {e}")
            self.df = pd.read_csv(self.data_file_path, index_col='Date', parse_dates=True)

        # Standardize column access. If columns are MultiIndex (e.g., ('Close', 'AAPL')),
        # extract the ticker's data to simplify to single-level columns.
        if isinstance(self.df.columns, pd.MultiIndex):
            # Expected levels are often ('Price', 'Ticker') or similar from yfinance
            # If the ticker is part of the column name (e.g. ('Close', 'AAPL'))
            # we want to select columns for self.ticker and drop the ticker level.
            # Example: self.df.xs(self.ticker, level='Ticker', axis=1)
            # For simplicity, if it's ('Close', 'TICKER'), ('Open', 'TICKER'), etc.
            # we can just take the first level of the column names.
            # However, yfinance output seems to be (Value, Ticker), e.g. ('Close', 'AAPL')
            # So we need to select where second level of multiindex is self.ticker
            
            # Let's try to select the columns for the current ticker.
            # Assuming the structure is (Value, TickerSymbol) e.g. ('Close', 'AAPL')
            # We want to change columns from ('Close', 'AAPL') to 'Close'.
            # We can check if the ticker is present in the second level of the multi-index
            if self.ticker in self.df.columns.get_level_values(1):
                 self.df = self.df.xs(self.ticker, level=1, axis=1)
            else:
                # If the ticker is not in the second level, maybe it's a simpler structure
                # or a different multi-index. For now, let's assume this is an issue.
                print(f"Warning: Could not find ticker {self.ticker} in second level of MultiIndex columns. Columns: {self.df.columns}")
                # As a fallback, try to use the first level if it contains OHLCV
                if all(col_name in self.df.columns.get_level_values(0) for col_name in ['Open', 'High', 'Low', 'Close', 'Volume']):
                    self.df.columns = self.df.columns.get_level_values(0) # This might lose ticker info if multiple tickers were in CSV
                else:
                    raise ValueError(f"Cannot simplify MultiIndex columns for ticker {self.ticker}. Columns: {self.df.columns}")


        # Ensure Date index is datetime
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)

        # Preprocess data: add moving averages
        self.df['SMA5'] = self.df['Close'].rolling(window=5).mean()
        self.df['SMA20'] = self.df['Close'].rolling(window=self.window_size).mean()
        self.df.dropna(inplace=True) # Remove rows with NaN MA values

        # Filter data based on start_date_str and time_horizon_days
        # The actual start date for trading will be the first date in df >= start_date_str
        self.start_date_dt = pd.to_datetime(start_date_str)
        self.trade_df = self.df[self.df.index >= self.start_date_dt].copy()

        if len(self.trade_df) < self.time_horizon_days:
            # Not enough data for the full horizon from the specified start date
            # This might be an issue if we strictly need time_horizon_days
            # For now, we'll just use the available data
            pass # Or raise an error: raise ValueError("Not enough data for the given start_date and time_horizon")
        
        if self.trade_df.empty:
            raise ValueError(f"No trading data available for {self.ticker} starting from {self.start_date_str}. "
                             f"Oldest data point is {self.df.index.min()}, newest is {self.df.index.max()}")

        # Action space: Box space for number of shares to trade.
        # Positive: buy, Negative: sell. For simplicity, let's set a max trade amount.
        # Max shares to trade in one step (e.g., based on a fraction of typical daily volume, or just a large number)
        # Let's use a large number like 1,000,000. Agent can trade fractional shares.
        self.max_trade_shares = 1_000_000 
        self.action_space = spaces.Box(low=-self.max_trade_shares, high=self.max_trade_shares, shape=(1,), dtype=np.float32)

        # Observation space: [current_cash, shares_held, current_price, sma5, sma20]
        # Using np.finfo(np.float32).max for cash and shares might be too large.
        # Let's estimate reasonable max values.
        # Max cash could be initial_cash + (time_horizon_days * cash_inflow_per_step) + profit from trading (hard to bound)
        # Max shares: if all cash is used to buy shares at $1. Let's use a large practical number.
        # Max price: From data or a large number.
        
        max_price_in_data = self.df['Close'].max() if not self.df.empty else 10000
        
        obs_low = np.array([0, 0, 0, 0, 0], dtype=np.float32) # Cash, Shares, Price, SMA5, SMA20
        obs_high = np.array([np.finfo(np.float32).max, # Cash can grow indefinitely
                             np.finfo(np.float32).max, # Shares can grow indefinitely theoretically
                             max_price_in_data * 5,    # Allow for price increases
                             max_price_in_data * 5,
                             max_price_in_data * 5], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Episode state
        self.current_step = 0
        self.current_cash = 0
        self.shares_held = 0
        self.current_date_idx = 0 # Index within self.trade_df
        self._current_data_row = None # To store the pd.Series for the current day

        # For rendering
        self.fig = None
        self.ax = None
        self.trade_history = [] # Store (day_index, action_type, price, num_shares)

    def _get_obs(self):
        if self._current_data_row is None: # Should not happen if reset is called first
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        obs = np.array([
            self.current_cash,
            self.shares_held,
            self._current_data_row['Close'],
            self._current_data_row['SMA5'],
            self._current_data_row['SMA20']
        ], dtype=np.float32)
        return obs

    def _get_info(self):
        current_price_for_info = 0
        date_for_info_str = "N/A"
        
        if self._current_data_row is not None:
            current_price_for_info = self._current_data_row['Close']
            date_for_info_str = self._current_data_row.name.strftime("%Y-%m-%d")
        elif self.current_date_idx > 0 and self.current_date_idx <= len(self.trade_df) and not self.trade_df.empty:
            # This case handles when _current_data_row might be None (e.g. after termination and advancing idx out of bounds),
            # but we still want to report info for the last valid day.
            # The step function already tries to set _current_data_row to the last valid day on termination.
            last_valid_idx = min(self.current_date_idx - 1, len(self.trade_df) - 1)
            if last_valid_idx >= 0:
                last_data_row_for_info = self.trade_df.iloc[last_valid_idx]
                current_price_for_info = last_data_row_for_info['Close']
                date_for_info_str = last_data_row_for_info.name.strftime("%Y-%m-%d")

        portfolio_value = self.current_cash + (self.shares_held * current_price_for_info)
        
        return {
            "current_date": date_for_info_str,
            "current_price": current_price_for_info,
            "shares_held": self.shares_held,
            "current_cash": self.current_cash,
            "portfolio_value": portfolio_value
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_cash = self.initial_cash
        self.shares_held = 0.0
        self.current_step = 0
        
        # Find the first valid trading day in trade_df (it's already filtered by start_date)
        self.current_date_idx = 0 
        if self.current_date_idx >= len(self.trade_df):
             raise ValueError(f"No valid trading days found in trade_df after filtering for start date {self.start_date_str} "
                              f"and dropping NaNs from MAs. Check data availability and MA window size. "
                              f"trade_df length: {len(self.trade_df)}, df length: {len(self.df)}")

        self._current_data_row = self.trade_df.iloc[self.current_date_idx]
        
        self.trade_history = []

        if self.fig is not None and self.render_mode == 'human':
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            
        return self._get_obs(), self._get_info()

    def step(self, action):
        action_shares = action[0] # Action is a single float value: shares to trade
        current_price = self._current_data_row['Close']
        
        # Store cash before trade for reward calculation
        cash_before_trade = self.current_cash

        # Execute trade
        if action_shares > 0: # Buy
            shares_to_buy = action_shares
            cost = shares_to_buy * current_price
            if self.current_cash >= cost:
                self.current_cash -= cost
                self.shares_held += shares_to_buy
                self.trade_history.append({'date_idx': self.current_date_idx, 'type': 'BUY', 'price': current_price, 'shares': shares_to_buy})
            else:
                pass # No trade if not enough cash for the full requested amount

        elif action_shares < 0: # Sell
            shares_to_sell = abs(action_shares)
            if self.shares_held >= shares_to_sell:
                self.current_cash += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                self.trade_history.append({'date_idx': self.current_date_idx, 'type': 'SELL', 'price': current_price, 'shares': shares_to_sell})
            else:
                pass # No trade if not enough shares for the full requested amount
        
        # Reward: change in cash from trading for this step
        reward_from_trade = self.current_cash - cash_before_trade
        
        # Add cash inflow for the step (after trade and reward calculation for trade)
        self.current_cash += self.cash_inflow_per_step

        # Move to next day state
        self.current_step += 1
        self.current_date_idx += 1

        terminated = False
        truncated = False # Not used yet, but part of Gym API

        # Determine if episode is terminated
        if self.current_date_idx >= len(self.trade_df):
            terminated = True # End of available data
            # _current_data_row will be from the last valid day if we fetch it before this check,
            # or None if we try to fetch after current_date_idx is out of bounds.
            # For terminal reward, we need the price from the day the episode ends.
            # So, _current_data_row should be the one from current_date_idx-1 (the day that just completed)
            if self.current_date_idx > 0:
                 self._current_data_row = self.trade_df.iloc[self.current_date_idx -1]
            else: # Should not happen if reset properly initializes.
                 self._current_data_row = None

        elif self.current_step >= self.time_horizon_days:
            terminated = True # Reached time horizon
            # The current day's data (self.trade_df.iloc[self.current_date_idx-1]) is already set
            # from the end of the previous step, or start of this step.
            # This self._current_data_row is the one whose price we use.
            if self.current_date_idx > 0:
                 self._current_data_row = self.trade_df.iloc[self.current_date_idx-1] # Data for the day that just finished
            else:
                 self._current_data_row = None


        reward = reward_from_trade
        if terminated and self._current_data_row is not None:
            current_price_at_termination = self._current_data_row['Close']
            value_of_held_shares = self.shares_held * current_price_at_termination
            reward += value_of_held_shares
        
        # Prepare next observation
        if not terminated:
            if self.current_date_idx < len(self.trade_df):
                self._current_data_row = self.trade_df.iloc[self.current_date_idx]
            else: # Should be caught by termination, but as a safeguard
                self._current_data_row = None 
                terminated = True # Ensure termination if somehow missed
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self.render()
            
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'ansi':
            info = self._get_info()
            total_cash_injected = self.initial_cash + (self.current_step * self.cash_inflow_per_step)
            output = (f"Step: {self.current_step}, Date: {info['current_date']}, Price: {info['current_price']:.2f}\n"
                      f"  Shares: {self.shares_held:.2f}, Cash: {self.current_cash:.2f}, Portfolio: {info['portfolio_value']:.2f}\n"
                      f"  Total Cash Injected: {total_cash_injected:.2f}")
            if self.trade_history:
                last_trade = self.trade_history[-1]
                # Check if last trade was on the previous day (current_date_idx would have advanced from the step where trade occurred)
                # The trade_history stores date_idx which is the index in trade_df for the day of the trade.
                # self.current_step is 1-based for user display, self.current_date_idx is 0-based for trade_df.
                # If a trade happened on current_date_idx = k (which means step k+1 is starting or just finished),
                # then after the step, current_date_idx becomes k+1.
                # So, if last_trade['date_idx'] == self.current_date_idx -1, it means the trade happened in the step that just completed.
                if self.current_date_idx > 0 and last_trade['date_idx'] == (self.current_date_idx -1):
                     output += f", Last Action: {last_trade['type']} {last_trade['shares']:.2f} @ {last_trade['price']:.2f}"
            print(output)
            return output

        if self.render_mode == 'human':
            if self._current_data_row is None and self.current_date_idx > 0: # End of episode
                current_plot_date_idx = self.current_date_idx -1
                current_actual_date = self.trade_df.index[current_plot_date_idx]
            elif self._current_data_row is not None:
                current_plot_date_idx = self.current_date_idx
                current_actual_date = self._current_data_row.name # This is the date index
            else: # Should not happen if reset is called
                return


            if self.fig is None:
                self.fig, self.ax = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
                plt.ion() # Interactive mode on

            self.ax[0].clear()
            self.ax[1].clear()

            # Determine plot window
            start_plot_idx = max(0, current_plot_date_idx - self.render_lookback_window)
            # Ensure end_plot_idx does not exceed available data, even if current_plot_date_idx is near the end
            end_plot_idx = min(len(self.trade_df) -1 , current_plot_date_idx + self.render_lookback_window // 2) 
            
            # Ensure start_plot_idx is not greater than end_plot_idx if data is short
            if start_plot_idx > end_plot_idx and len(self.trade_df) > 0:
                start_plot_idx = 0
                end_plot_idx = len(self.trade_df) -1


            if start_plot_idx <= end_plot_idx :
                plot_df = self.trade_df.iloc[start_plot_idx : end_plot_idx + 1]

                self.ax[0].plot(plot_df.index, plot_df['Close'], label='Close Price', color='blue')
                self.ax[0].plot(plot_df.index, plot_df['SMA5'], label='SMA5', color='orange', linestyle='--')
                self.ax[0].plot(plot_df.index, plot_df['SMA20'], label='SMA20', color='green', linestyle='--')

                # Plot trades within this window
                buys_x, buys_y, sells_x, sells_y = [], [], [], []
                for trade in self.trade_history:
                    trade_date = self.trade_df.index[trade['date_idx']]
                    if plot_df.index.min() <= trade_date <= plot_df.index.max():
                        if trade['type'].startswith('BUY'):
                            buys_x.append(trade_date)
                            buys_y.append(trade['price'])
                        elif trade['type'].startswith('SELL'):
                            sells_x.append(trade_date)
                            sells_y.append(trade['price'])
                
                self.ax[0].scatter(buys_x, buys_y, marker='^', color='green', s=100, label='Buy')
                self.ax[0].scatter(sells_x, sells_y, marker='v', color='red', s=100, label='Sell')
            
            self.ax[0].axvline(current_actual_date, color='gray', linestyle=':', lw=1, label='Current Day')
            self.ax[0].set_title(f"{self.ticker} Trading - Day {self.current_step} ({current_actual_date.strftime('%Y-%m-%d')})")
            self.ax[0].set_xlabel("Date")
            self.ax[0].set_ylabel("Price")
            self.ax[0].legend(loc='upper left')
            self.ax[0].grid(True)

            # Display portfolio info
            info = self._get_info()
            total_cash_injected = self.initial_cash + (self.current_step * self.cash_inflow_per_step)
            info_text = (f"Cash: ${self.current_cash:,.2f} | Shares: {self.shares_held:.2f}\n"
                         f"Portfolio Value: ${info['portfolio_value']:,.2f} | Current Price: ${info['current_price']:.2f}\n"
                         f"Total Cash Injected: ${total_cash_injected:,.2f}")
            
            # Check if a trade occurred in the step that just completed.
            # self.current_date_idx has been incremented for the *next* step.
            # So, the trade would have occurred on self.current_date_idx - 1.
            if self.trade_history:
                last_trade = self.trade_history[-1]
                if self.current_date_idx > 0 and last_trade['date_idx'] == (self.current_date_idx - 1):
                     info_text += f"\nLast Action: {last_trade['type']} {last_trade['shares']:.2f} @ ${last_trade['price']:.2f}"
            
            self.ax[1].text(0.5, 0.5, info_text, horizontalalignment='center', verticalalignment='center', fontsize=11, transform=self.ax[1].transAxes)
            self.ax[1].axis('off')
            
            plt.tight_layout()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            if self.metadata['render_fps'] > 0:
                 plt.pause(1.0 / self.metadata['render_fps'])


    def close(self):
        if self.fig is not None:
            plt.ioff() # Turn off interactive mode
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        print("Trading environment closed.")
